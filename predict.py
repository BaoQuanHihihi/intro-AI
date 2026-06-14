#!/usr/bin/env python3
"""Inference on test set -> predictions.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    ASPECT_COLS,
    ReviewAspectDataset,
    find_id_column,
    load_test_frame,
    resolve_sample_submission_path,
    resolve_test_path,
)
from src.model import JointAspectSentimentModel
from src.utils import setup_logging, try_cuda_device

LOG = setup_logging()


def _safe_stt(x) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("stt/id không hợp lệ")
    return int(float(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Thư mục chứa best_model.pt + tokenizer/")
    p.add_argument("--test_path", type=str, default=None)
    p.add_argument("--sample_submission_path", type=str, default=None)
    p.add_argument("--output_csv", type=str, default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def load_checkpoint_model(checkpoint_dir: Path, device: torch.device):
    ckpt_path = checkpoint_dir / "best_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt["model_name"]
    model = JointAspectSentimentModel(model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tok_dir = checkpoint_dir / "tokenizer"
    if not tok_dir.is_dir():
        LOG.warning("Không có %s — tải tokenizer từ %s", tok_dir, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

    meta = checkpoint_dir / "training_meta.json"
    max_length = 256
    if meta.is_file():
        import json

        with meta.open("r", encoding="utf-8") as f:
            m = json.load(f)
        max_length = int(m.get("max_length", 256))
    return model, tokenizer, max_length


@torch.no_grad()
def predict_batches(
    model: JointAspectSentimentModel,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids, mask)
        pred = logits.argmax(dim=-1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def main() -> None:
    args = parse_args()
    workspace = ROOT
    device = try_cuda_device()
    LOG.info("Device: %s", device)

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    test_path = resolve_test_path(workspace, args.test_path)
    LOG.info("Test data: %s", test_path)

    sample_path = resolve_sample_submission_path(workspace, args.sample_submission_path)

    test_df = load_test_frame(test_path)
    id_col = find_id_column(test_df, required=False)

    model, tokenizer, max_length = load_checkpoint_model(ckpt_dir, device)

    if sample_path is not None and sample_path.is_file():
        sub = pd.read_csv(sample_path, encoding="utf-8")
        if "stt" not in sub.columns:
            raise ValueError("sample_submission phải có cột stt")
        order_ids = sub["stt"].tolist()
        if id_col is None:
            raise ValueError(
                "Có sample_submission nhưng test không có cột stt/id để ghép. "
                "Thêm cột stt vào test hoặc bỏ sample_submission."
            )
        key = test_df[id_col].map(_safe_stt)
        id_to_idx = {k: i for i, k in enumerate(key)}
        missing = [_safe_stt(i) for i in order_ids if _safe_stt(i) not in id_to_idx]
        if missing:
            raise ValueError(f"sample_submission có stt không có trong test (ví dụ {missing[:5]})")
        indices = [id_to_idx[_safe_stt(s)] for s in order_ids]
        infer_df = test_df.iloc[indices].reset_index(drop=True)
        out_stt = [_safe_stt(x) for x in order_ids]
    else:
        infer_df = test_df
        if id_col is not None:
            out_stt = [_safe_stt(x) for x in infer_df[id_col].tolist()]
        else:
            out_stt = list(range(1, len(infer_df) + 1))
            LOG.warning("Test không có stt/id — dùng 1..N làm stt trong predictions.csv")

    ds = ReviewAspectDataset(infer_df["_text"].tolist(), tokenizer, max_length, label_matrix=None)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    pred_mat = predict_batches(model, loader, device)

    if pred_mat.shape != (len(infer_df), 6):
        raise RuntimeError(f"Shape pred sai: {pred_mat.shape}, kỳ vọng ({len(infer_df)}, 6)")

    if (pred_mat < 0).any() or (pred_mat > 5).any():
        raise ValueError("Dự đoán ngoài [0,5]")

    out = pd.DataFrame(pred_mat, columns=list(ASPECT_COLS), dtype=np.int64)
    out.insert(0, "stt", out_stt)
    out_path = Path(args.output_csv).expanduser()
    if not out_path.is_absolute():
        out_path = workspace / out_path
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    LOG.info("Đã ghi %s (%d dòng)", out_path, len(out))


if __name__ == "__main__":
    main()
