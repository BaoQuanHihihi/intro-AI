#!/usr/bin/env python3
"""Train joint aspect-sentiment model with validation early stopping."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# repo root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TrainConfig, build_train_argparser, load_yaml_config, merge_cli_into_config
from src.data import (
    ASPECT_COLS,
    ReviewAspectDataset,
    compute_aspect_class_weights,
    load_train_val_frames,
    resolve_train_path,
)
from src.metrics import compute_all_metrics, labels_to_numpy
from src.model import JointAspectSentimentModel, build_model_and_tokenizer
from src.utils import ensure_dir, save_json, set_seed, setup_logging, try_cuda_device

LOG = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = build_train_argparser()
    return parser.parse_args()


def get_config(ns: argparse.Namespace) -> TrainConfig:
    raw = load_yaml_config(Path(ns.config))
    train_section = raw.get("train", raw)
    cfg = TrainConfig.from_dict(train_section if isinstance(train_section, dict) else {})
    return merge_cli_into_config(cfg, ns)


def set_encoder_trainable(model: JointAspectSentimentModel, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def evaluate_epoch(
    model: JointAspectSentimentModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, mask)
        pred = logits.argmax(dim=-1)
        all_pred.append(labels_to_numpy(pred))
        all_true.append(labels_to_numpy(labels))
    y_p = np.concatenate(all_pred, axis=0)
    y_t = np.concatenate(all_true, axis=0)
    return compute_all_metrics(y_t, y_p)


def train() -> None:
    ns = parse_args()
    cfg = get_config(ns)
    set_seed(cfg.seed)
    device = try_cuda_device()
    LOG.info("Device: %s", device)

    workspace = ROOT
    train_path = resolve_train_path(workspace, cfg.train_path)
    LOG.info("Train data: %s", train_path)

    train_df, val_df = load_train_val_frames(train_path, cfg.val_ratio, cfg.seed)
    LOG.info("Train rows: %d | Val rows: %d", len(train_df), len(val_df))

    y_train = train_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    class_weights = compute_aspect_class_weights(y_train) if cfg.use_class_weights else None

    model, tokenizer, name_used = build_model_and_tokenizer(cfg.model_name, cfg.fallback_model_name)
    model.to(device)

    train_ds = ReviewAspectDataset(
        train_df["_text"].tolist(),
        tokenizer,
        cfg.max_length,
        label_matrix=y_train,
    )
    val_ds = ReviewAspectDataset(
        val_df["_text"].tolist(),
        tokenizer,
        cfg.max_length,
        label_matrix=val_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    criteria: list[nn.CrossEntropyLoss] = []
    for a in range(6):
        w = class_weights[a].to(device) if class_weights is not None else None
        criteria.append(
            nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
        )

    no_decay = ["bias", "LayerNorm.weight"]
    enc_params = list(model.encoder.named_parameters())
    head_params = list(model.heads.named_parameters())

    def grouped_params():
        groups = []
        enc_decay = [p for n, p in enc_params if not any(nd in n for nd in no_decay)]
        enc_no_decay = [p for n, p in enc_params if any(nd in n for nd in no_decay)]
        head_decay = [p for n, p in head_params if not any(nd in n for nd in no_decay)]
        head_no_decay = [p for n, p in head_params if any(nd in n for nd in no_decay)]
        if enc_decay:
            groups.append({"params": enc_decay, "weight_decay": cfg.weight_decay})
        if enc_no_decay:
            groups.append({"params": enc_no_decay, "weight_decay": 0.0})
        if head_decay:
            groups.append({"params": head_decay, "weight_decay": cfg.weight_decay})
        if head_no_decay:
            groups.append({"params": head_no_decay, "weight_decay": 0.0})
        return groups

    optim = AdamW(grouped_params(), lr=cfg.learning_rate, eps=1e-8)

    num_update_steps = math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.num_epochs
    warmup_steps = int(num_update_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    out_dir = ensure_dir(Path(cfg.output_dir).resolve())
    tok_dir = ensure_dir(out_dir / "tokenizer")
    tokenizer.save_pretrained(tok_dir)

    meta = {
        "model_name": name_used,
        "requested_model_name": cfg.model_name,
        "fallback_model_name": cfg.fallback_model_name,
        "max_length": cfg.max_length,
        "aspect_cols": list(ASPECT_COLS),
        "train_config": cfg.to_dict(),
    }
    save_json(out_dir / "training_meta.json", meta)

    best_overall = -1.0
    best_epoch = -1
    patience_left = cfg.early_stopping_patience

    for epoch in range(1, cfg.num_epochs + 1):
        if cfg.freeze_encoder_epochs > 0 and epoch <= cfg.freeze_encoder_epochs:
            set_encoder_trainable(model, False)
        else:
            set_encoder_trainable(model, True)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        accum = 0
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask)
            loss_parts = []
            for a in range(6):
                loss_parts.append(criteria[a](logits[:, a, :], labels[:, a]))
            loss = torch.stack(loss_parts).mean()
            (loss / cfg.grad_accum_steps).backward()
            accum += 1

            is_last = step == len(train_loader) - 1
            if accum % cfg.grad_accum_steps == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                accum = 0

            epoch_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)
        metrics = evaluate_epoch(model, val_loader, device)
        LOG.info(
            "Epoch %d | train_loss=%.5f | val_micro_f1=%.4f | val_sentiment_acc=%.4f | val_overall_score=%.4f",
            epoch,
            avg_train_loss,
            metrics["micro_f1"],
            metrics["sentiment_accuracy"],
            metrics["overall_score"],
        )

        if metrics["overall_score"] > best_overall:
            best_overall = metrics["overall_score"]
            best_epoch = epoch
            patience_left = cfg.early_stopping_patience
            ckpt_path = out_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": name_used,
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "config": cfg.to_dict(),
                },
                ckpt_path,
            )
            LOG.info("Saved best checkpoint: %s (overall=%.4f)", ckpt_path, best_overall)
        else:
            patience_left -= 1
            LOG.info("No improvement. Patience: %d/%d", patience_left, cfg.early_stopping_patience)
            if patience_left <= 0:
                LOG.info("Early stopping.")
                break

    LOG.info(
        "Training done. Best epoch=%d | best val_overall_score=%.4f | output_dir=%s",
        best_epoch,
        best_overall,
        out_dir,
    )


if __name__ == "__main__":
    train()
