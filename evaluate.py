#!/usr/bin/env python3
"""Tính micro-F1, sentiment accuracy, overall từ file CSV hoặc model trên tập val."""

from __future__ import annotations

import argparse
import json
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
    load_train_val_frames,
    resolve_train_path,
    validate_label_columns,
)
from src.metrics import compute_all_metrics
from src.model import JointAspectSentimentModel
from src.utils import set_seed, setup_logging, try_cuda_device

LOG = setup_logging()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("from_files", help="So sánh predictions.csv với ground truth CSV")
    f.add_argument("--pred_path", type=str, required=True)
    f.add_argument("--gt_path", type=str, required=True)

    m = sub.add_parser("from_model", help="Chạy checkpoint trên validation split từ train")
    m.add_argument("--checkpoint_dir", type=str, required=True)
    m.add_argument("--train_path", type=str, default=None)
    m.add_argument("--val_ratio", type=float, default=0.1)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--batch_size", type=int, default=32)

    return p.parse_args()


def align_pred_gt(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    for c in ["stt", *ASPECT_COLS]:
        if c not in pred_df.columns:
            raise ValueError(f"pred thiếu cột {c}. Có: {list(pred_df.columns)}")
    validate_label_columns(gt_df)

    gt_id = find_id_column(gt_df, required=False)
    if "stt" in pred_df.columns and gt_id is not None:
        pr = pred_df.drop_duplicates("stt").set_index("stt")
        gt = gt_df.drop_duplicates(gt_id).set_index(gt_id)
        common = pr.index.intersection(gt.index)
        if len(common) == 0:
            raise ValueError("Không có id/stt chung giữa pred và gt")
        common = sorted(common)
        y_p = pr.loc[common, list(ASPECT_COLS)].to_numpy(dtype=np.int64)
        y_t = gt.loc[common, list(ASPECT_COLS)].to_numpy(dtype=np.int64)
        LOG.info("Đã ghép theo stt/id: %d dòng", len(common))
        return y_t, y_p

    if len(pred_df) != len(gt_df):
        raise ValueError(
            "Không có cột stt/id để ghép mà len(pred) != len(gt): "
            f"{len(pred_df)} vs {len(gt_df)}"
        )
    y_p = pred_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    y_t = gt_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    return y_t, y_p


def eval_from_files(pred_path: Path, gt_path: Path) -> dict[str, float]:
    pred_df = pd.read_csv(pred_path, encoding="utf-8")
    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    y_t, y_p = align_pred_gt(pred_df, gt_df)
    return compute_all_metrics(y_t, y_p)


def load_ckpt(ckpt_dir: Path, device: torch.device):
    ckpt_path = ckpt_dir / "best_model.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt["model_name"]
    model = JointAspectSentimentModel(model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    tok_dir = ckpt_dir / "tokenizer"
    if tok_dir.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    meta = ckpt_dir / "training_meta.json"
    max_length = 256
    if meta.is_file():
        with meta.open("r", encoding="utf-8") as f:
            max_length = int(json.load(f).get("max_length", 256))
    return model, tokenizer, max_length


@torch.no_grad()
def eval_from_model(
    ckpt_dir: Path,
    train_path: Path,
    val_ratio: float,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    _, val_df = load_train_val_frames(train_path, val_ratio, seed)
    y_true = val_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    model, tokenizer, max_length = load_ckpt(ckpt_dir, device)
    ds = ReviewAspectDataset(val_df["_text"].tolist(), tokenizer, max_length, label_matrix=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Val inference"):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    return compute_all_metrics(y_true, y_pred)


def main() -> None:
    args = parse_args()
    if args.mode == "from_files":
        m = eval_from_files(Path(args.pred_path), Path(args.gt_path))
    else:
        workspace = ROOT
        train_path = resolve_train_path(workspace, getattr(args, "train_path", None))
        device = try_cuda_device()
        m = eval_from_model(
            Path(args.checkpoint_dir).resolve(),
            train_path,
            args.val_ratio,
            args.seed,
            args.batch_size,
            device,
        )

    LOG.info("micro_f1:           %.6f", m["micro_f1"])
    LOG.info("sentiment_accuracy:  %.6f", m["sentiment_accuracy"])
    LOG.info("overall_score:       %.6f", m["overall_score"])
    print(
        json.dumps(
            {
                "micro_f1": m["micro_f1"],
                "sentiment_accuracy": m["sentiment_accuracy"],
                "overall_score": m["overall_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
