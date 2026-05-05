#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import ArgoverseV2Dataset
from datasets import InteractionDIGIRDataset
from predictors import QCNet
from transforms import TargetBuilder


def parse_model_specs(specs: Iterable[str]) -> List[Tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be NAME=/path/to.ckpt, got {spec!r}")
        name, ckpt = spec.split("=", 1)
        parsed.append((name.strip(), ckpt.strip()))
    return parsed


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value.item()) if value.numel() == 1 else value.tolist()
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def build_dataset(args, model: QCNet):
    if model.dataset == "interaction_digir":
        if args.interaction_data_path is None:
            raise ValueError("--interaction_data_path is required for interaction_digir validation")
        return InteractionDIGIRDataset(
            data_path=args.interaction_data_path,
            split=args.split,
            transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
            num_historical_steps=model.num_historical_steps,
            num_future_steps=model.num_future_steps,
            max_samples=args.max_samples,
            use_kg=args.use_kg,
            allow_test_as_val=args.allow_test_as_val,
            locations=[x.strip() for x in args.locations.split(",") if x.strip()] if args.locations else None,
        )
    if model.dataset == "argoverse_v2":
        return ArgoverseV2Dataset(
            root=args.root,
            split=args.split,
            transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        )
    raise ValueError(f"{model.dataset} is not a supported dataset")


def validate_one(args, name: str, ckpt_path: str) -> Dict[str, object]:
    print(f"validating {name}: {ckpt_path}", flush=True)
    model = QCNet.load_from_checkpoint(checkpoint_path=ckpt_path, map_location="cpu")
    dataset = build_dataset(args, model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=args.progress_bar,
    )
    metrics = trainer.validate(model, loader, verbose=False, ckpt_path=None)[0]
    row = {"model": name, "checkpoint": ckpt_path}
    row.update({key: tensor_to_float(value) for key, value in metrics.items()})
    return row


def write_outputs(rows: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "qcnet_val_metrics.json"
    csv_path = out_dir / "qcnet_val_metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QCNet.validate on one or more checkpoints and save the exact validation metrics."
    )
    parser.add_argument("--model", action="append", required=True, help="NAME=/path/to/checkpoint.ckpt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--interaction_data_path", default=None)
    parser.add_argument("--locations", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_kg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_test_as_val", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--progress_bar", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(2023, workers=True)
    rows = [validate_one(args, name, ckpt) for name, ckpt in parse_model_specs(args.model)]
    write_outputs(rows, Path(args.out_dir))


if __name__ == "__main__":
    main()
