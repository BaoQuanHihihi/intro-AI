"""Load and merge YAML config with CLI overrides."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TrainConfig:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 15
    warmup_ratio: float = 0.06
    seed: int = 42
    val_ratio: float = 0.1
    early_stopping_patience: int = 3
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    sample_submission_path: Optional[str] = None
    output_dir: str = "outputs/run"
    freeze_encoder_epochs: int = 0
    fallback_model_name: str = "xlm-roberta-base"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
        }


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def build_train_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train joint aspect model")
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, dest="learning_rate")
    p.add_argument("--epochs", type=int, default=None, dest="num_epochs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=None)
    return p


def merge_cli_into_config(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    d = cfg.to_dict()
    if args.model_name is not None:
        d["model_name"] = args.model_name
    if args.max_length is not None:
        d["max_length"] = args.max_length
    if args.batch_size is not None:
        d["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        d["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        d["num_epochs"] = args.num_epochs
    if args.seed is not None:
        d["seed"] = args.seed
    if args.output_dir is not None:
        d["output_dir"] = args.output_dir
    if args.train_path is not None:
        d["train_path"] = args.train_path
    if args.val_ratio is not None:
        d["val_ratio"] = args.val_ratio
    if args.early_stopping_patience is not None:
        d["early_stopping_patience"] = args.early_stopping_patience
    return TrainConfig.from_dict(d)


def get_train_config(args: Optional[argparse.Namespace] = None) -> TrainConfig:
    parser = build_train_argparser()
    ns = parser.parse_args(args)
    raw = load_yaml_config(Path(ns.config))
    train_section = raw.get("train", raw)
    cfg = TrainConfig.from_dict(train_section if isinstance(train_section, dict) else {})
    return merge_cli_into_config(cfg, ns)
