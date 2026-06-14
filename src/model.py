"""Joint encoder + 6 aspect heads (6 classes each: 0..5)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

LOG = logging.getLogger("review_analytics")


class JointAspectSentimentModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int = 6, num_classes: int = 6) -> None:
        super().__init__()
        self.backbone_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(num_aspects)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS
        pooled = out.last_hidden_state[:, 0, :]
        logits = torch.stack([h(pooled) for h in self.heads], dim=1)
        # [batch, 6 aspects, 6 classes]
        return logits

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)


def build_model_and_tokenizer(
    model_name: str,
    fallback_model_name: str,
) -> Tuple[JointAspectSentimentModel, object, str]:
    """Try primary HF name, then fallback. Returns (model, tokenizer, name_used)."""
    last_err: Optional[BaseException] = None
    for name in (model_name, fallback_model_name):
        if not name:
            continue
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
            model = JointAspectSentimentModel(name)
            if name != model_name:
                LOG.warning("Dùng fallback encoder: %s (lỗi khi tải %s)", name, model_name)
            return model, tokenizer, name
        except Exception as e:
            last_err = e
            LOG.warning("Không tải được %s: %s", name, e)
    raise RuntimeError(f"Không tải được encoder. Lỗi cuối: {last_err}") from last_err
