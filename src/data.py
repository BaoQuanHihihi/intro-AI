"""Data loading, path resolution, preprocessing, and PyTorch Dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.metrics import ASPECT_COLS

LOG = logging.getLogger("review_analytics")

TEXT_CANDIDATES = ("review", "Review", "text", "content", "comment", "noi_dung", "noidung")
ID_CANDIDATES = ("stt", "id", "ID", "idx", "index")

TRAIN_FILE_CANDIDATES = (
    "data/train.csv",
    "data/train-problem.csv",
    "train.csv",
    "data/train_problem.csv",
)

TEST_FILE_CANDIDATES = (
    "data/test.csv",
    "data/gt_reviews_test.csv",
    "test.csv",
    "data/public_test.csv",
)

SAMPLE_SUBMISSION_CANDIDATES = (
    "data/sample_submission.csv",
    "sample_submission.csv",
    "data/sample-submission.csv",
)


def _first_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for rel in candidates:
        p = base / rel if not Path(rel).is_absolute() else Path(rel)
        if p.is_file():
            return p.resolve()
    return None


def resolve_train_path(workspace: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"train_path not found: {p}")
        return p.resolve()
    found = _first_existing(workspace, TRAIN_FILE_CANDIDATES)
    if found:
        return found
    raise FileNotFoundError(
        "Không tìm thấy file train. Truyền --train_path hoặc đặt một trong: "
        + ", ".join(TRAIN_FILE_CANDIDATES)
    )


def resolve_test_path(workspace: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"test_path not found: {p}")
        return p.resolve()
    found = _first_existing(workspace, TEST_FILE_CANDIDATES)
    if found:
        return found
    raise FileNotFoundError(
        "Không tìm thấy file test. Truyền --test_path hoặc đặt một trong: "
        + ", ".join(TEST_FILE_CANDIDATES)
    )


def resolve_sample_submission_path(workspace: Path, override: Optional[str]) -> Optional[Path]:
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"sample_submission_path not found: {p}")
        return p.resolve()
    found = _first_existing(workspace, SAMPLE_SUBMISSION_CANDIDATES)
    return found


def find_text_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in TEXT_CANDIDATES:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    raise ValueError(
        "Không tìm thấy cột text. Kỳ vọng một trong: "
        + ", ".join(TEXT_CANDIDATES)
        + f". Có trong file: {list(df.columns)}"
    )


def find_id_column(df: pd.DataFrame, required: bool = False) -> Optional[str]:
    for cand in ID_CANDIDATES:
        if cand in df.columns:
            return cand
    lower = {c.lower(): c for c in df.columns}
    for cand in ID_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Thiếu cột id (stt/id). Cột hiện có: {list(df.columns)}")
    return None


def validate_label_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ASPECT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột nhãn: {missing}. Cột hiện có: {list(df.columns)}")
    subset = df[list(ASPECT_COLS)]
    if subset.isnull().any().any():
        raise ValueError("Nhãn không được chứa NaN.")
    arr = subset.to_numpy(dtype=np.int64)
    if (arr < 0).any() or (arr > 5).any():
        bad = np.argwhere((arr < 0) | (arr > 5))
        raise ValueError(f"Nhãn phải trong [0,5]. Vi phạm tại các vị trí (row, col): {bad[:10]}...")


def normalize_text(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).replace("\r\n", "\n").strip()
    return t if t else " "


def _stratified_train_val_indices(
    strat: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-stratum split: ~val_ratio mỗi lớp; stratum 1 phần tử → toàn bộ train."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for s in np.unique(strat):
        idx = np.where(strat == s)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n <= 1:
            train_parts.append(idx)
            continue
        n_val = int(round(n * val_ratio))
        n_val = min(max(n_val, 1), n - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def load_train_val_frames(
    train_path: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(train_path, encoding="utf-8")
    validate_label_columns(df)
    text_col = find_text_column(df)
    df["_text"] = df[text_col].map(normalize_text)

    y_multi = df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    strat = (y_multi > 0).sum(axis=1)
    strat = strat.clip(0, 6)

    try:
        tr_idx, va_idx = _stratified_train_val_indices(strat, val_ratio, seed)
        if len(va_idx) == 0:
            raise ValueError("empty val")
        tr, va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    except Exception as e:
        LOG.warning(
            "Stratify thất bại (%s). Fallback: random split một lần — xem GUIDE.md.",
            e,
        )
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(df))
        n_val = max(1, int(round(len(df) * val_ratio)))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        tr, va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    return tr.reset_index(drop=True), va.reset_index(drop=True)


def load_train_full(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path, encoding="utf-8")
    validate_label_columns(df)
    text_col = find_text_column(df)
    df["_text"] = df[text_col].map(normalize_text)
    return df.reset_index(drop=True)


def load_test_frame(test_path: Path) -> pd.DataFrame:
    df = pd.read_csv(test_path, encoding="utf-8")
    text_col = find_text_column(df)
    df["_text"] = df[text_col].map(normalize_text)
    return df.reset_index(drop=True)


def compute_aspect_class_weights(
    labels: np.ndarray, num_classes: int = 6
) -> list[torch.Tensor]:
    """
    labels: [N, 6] int. Per-aspect inverse frequency weights, normalized to mean 1.
    """
    weights: list[torch.Tensor] = []
    for j in range(6):
        col = labels[:, j]
        counts = np.bincount(col, minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = 1.0 / counts
        w = w * (num_classes / w.sum())
        weights.append(torch.tensor(w, dtype=torch.float32))
    return weights


class ReviewAspectDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        max_length: int,
        label_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = label_matrix

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx] or " "
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
