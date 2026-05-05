#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import InteractionDIGIRDataset
from predictors import QCNet
from transforms import TargetBuilder


ENDPOINT_THRESHOLDS = (0.05, 0.10, 0.20, 0.50, 1.00)
TRAJ_THRESHOLDS = (0.05, 0.06, 0.08, 0.10)


def scene_type(location: str) -> str:
    low = location.lower()
    if "merging" in low:
        return "merge"
    if "intersection" in low:
        return "intersection"
    if "roundabout" in low:
        return "roundabout"
    return "other"


def make_bucket():
    return defaultdict(float)


def add_count(bucket, n: int) -> None:
    bucket["n"] += int(n)


def add_mean(bucket, key: str, values: torch.Tensor) -> None:
    values = values.detach().float()
    if values.numel() == 0:
        return
    bucket[f"{key}_sum"] += float(values.sum().cpu())
    bucket[f"{key}_n"] += int(values.numel())


def add_rate(bucket, key: str, values: torch.Tensor) -> None:
    values = values.detach().bool()
    if values.numel() == 0:
        return
    bucket[f"{key}_num"] += float(values.float().sum().cpu())
    bucket[f"{key}_den"] += int(values.numel())


def finalize_bucket(bucket) -> Dict[str, float]:
    out = {"n": int(bucket.get("n", 0))}
    for key, val in bucket.items():
        if key.endswith("_sum"):
            base = key[:-4]
            den = bucket.get(f"{base}_n", 0.0)
            out[base] = float(val / den) if den else 0.0
        elif key.endswith("_num"):
            base = key[:-4]
            den = bucket.get(f"{base}_den", 0.0)
            out[base] = float(val / den) if den else 0.0
    return out


def mode_ade(pred_xy: torch.Tensor, gt_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    dist = torch.norm(pred_xy - gt_xy.unsqueeze(1), dim=-1)
    valid = valid_mask.unsqueeze(1).to(dtype=dist.dtype)
    return (dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)


def mode_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    n, _, t, _ = pred_xy.shape
    step_ids = torch.arange(1, t + 1, device=pred_xy.device).view(1, t)
    last_idx = (valid_mask.to(dtype=step_ids.dtype) * step_ids).argmax(dim=-1)
    return torch.norm(
        pred_xy[torch.arange(n, device=pred_xy.device), :, last_idx] -
        gt_xy[torch.arange(n, device=pred_xy.device), last_idx].unsqueeze(1),
        dim=-1,
    )


def pairwise_mean_distance(xy: torch.Tensor) -> torch.Tensor:
    if xy.size(1) <= 1:
        return xy.new_zeros(xy.size(0))
    pair = torch.cdist(xy, xy)
    keep = torch.triu(torch.ones(xy.size(1), xy.size(1), device=xy.device, dtype=torch.bool), diagonal=1)
    return pair[:, keep].mean(dim=-1)


def nearest_cross_model_ade(src_xy: torch.Tensor,
                            dst_xy: torch.Tensor,
                            valid_mask: torch.Tensor) -> torch.Tensor:
    dist = torch.norm(src_xy[:, :, None] - dst_xy[:, None], dim=-1)
    valid = valid_mask[:, None, None].to(dtype=dist.dtype)
    pair_ade = (dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
    return pair_ade.min(dim=-1).values


def update_model(bucket,
                 prefix: str,
                 propose_ade: torch.Tensor,
                 propose_fde: torch.Tensor,
                 refine_ade: torch.Tensor,
                 refine_fde: torch.Tensor,
                 prob: torch.Tensor,
                 final_xy: torch.Tensor) -> None:
    idx = torch.arange(refine_ade.size(0), device=refine_ade.device)
    propose_best = propose_ade.argmin(dim=-1)
    refine_best = refine_ade.argmin(dim=-1)
    endpoint_best = refine_fde.argmin(dim=-1)
    top1 = prob.argmax(dim=-1)

    propose_oracle_ade = propose_ade[idx, propose_best]
    propose_oracle_fde = propose_fde[idx, propose_best]
    refine_oracle_ade = refine_ade[idx, refine_best]
    refine_oracle_fde = refine_fde[idx, refine_best]
    endpoint_oracle_fde = refine_fde[idx, endpoint_best]
    endpoint_mode_ade = refine_ade[idx, endpoint_best]
    top1_ade = refine_ade[idx, top1]
    top1_fde = refine_fde[idx, top1]

    add_mean(bucket, f"{prefix}_propose_oracle_ADE", propose_oracle_ade)
    add_mean(bucket, f"{prefix}_propose_oracle_FDE", propose_oracle_fde)
    add_mean(bucket, f"{prefix}_refine_oracle_ADE", refine_oracle_ade)
    add_mean(bucket, f"{prefix}_refine_oracle_FDE", refine_oracle_fde)
    add_mean(bucket, f"{prefix}_endpoint_oracle_FDE", endpoint_oracle_fde)
    add_mean(bucket, f"{prefix}_endpoint_best_mode_ADE", endpoint_mode_ade)
    add_mean(bucket, f"{prefix}_top1_ADE", top1_ade)
    add_mean(bucket, f"{prefix}_top1_FDE", top1_fde)
    add_mean(bucket, f"{prefix}_score_gap_ADE", top1_ade - refine_oracle_ade)
    add_mean(bucket, f"{prefix}_refine_gain_ADE", propose_oracle_ade - refine_oracle_ade)
    add_mean(bucket, f"{prefix}_refine_gain_FDE", propose_oracle_fde - refine_oracle_fde)
    add_mean(bucket, f"{prefix}_endpoint_to_traj_gap_ADE", endpoint_mode_ade - refine_oracle_ade)
    add_mean(bucket, f"{prefix}_final_endpoint_pairwise_mean", pairwise_mean_distance(final_xy))
    add_rate(bucket, f"{prefix}_top1_is_refine_oracle", top1 == refine_best)
    add_rate(bucket, f"{prefix}_endpoint_best_is_refine_best", endpoint_best == refine_best)

    for threshold in ENDPOINT_THRESHOLDS:
        name = str(threshold).replace(".", "p")
        add_rate(bucket, f"{prefix}_endpoint_oracle_FDE_le_{name}", endpoint_oracle_fde <= threshold)
    for threshold in TRAJ_THRESHOLDS:
        name = str(threshold).replace(".", "p")
        add_rate(bucket, f"{prefix}_refine_oracle_ADE_le_{name}", refine_oracle_ade <= threshold)


def update_student_teacher_gap(bucket,
                               student_refine_ade: torch.Tensor,
                               student_refine_fde: torch.Tensor,
                               teacher_refine_ade: torch.Tensor,
                               teacher_refine_fde: torch.Tensor,
                               student_xy: torch.Tensor,
                               teacher_xy: torch.Tensor,
                               valid_mask: torch.Tensor) -> None:
    idx = torch.arange(student_refine_ade.size(0), device=student_refine_ade.device)
    student_best = student_refine_ade.argmin(dim=-1)
    teacher_best = teacher_refine_ade.argmin(dim=-1)
    student_oracle_ade = student_refine_ade[idx, student_best]
    teacher_oracle_ade = teacher_refine_ade[idx, teacher_best]
    student_oracle_fde = student_refine_fde[idx, student_refine_fde.argmin(dim=-1)]
    teacher_oracle_fde = teacher_refine_fde[idx, teacher_refine_fde.argmin(dim=-1)]

    add_mean(bucket, "student_minus_teacher_refine_oracle_ADE", student_oracle_ade - teacher_oracle_ade)
    add_mean(bucket, "student_minus_teacher_endpoint_oracle_FDE", student_oracle_fde - teacher_oracle_fde)
    add_mean(bucket, "student_modes_to_nearest_teacher_ADE",
             nearest_cross_model_ade(student_xy, teacher_xy, valid_mask).mean(dim=-1))
    add_mean(bucket, "teacher_modes_to_nearest_student_ADE",
             nearest_cross_model_ade(teacher_xy, student_xy, valid_mask).mean(dim=-1))


@torch.no_grad()
def evaluate(args) -> Dict[str, Dict[str, float]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = InteractionDIGIRDataset(
        data_path=args.data,
        split=args.split,
        transform=TargetBuilder(8, 12),
        num_historical_steps=8,
        num_future_steps=12,
        max_samples=args.max_samples,
        use_kg=True,
        allow_test_as_val=args.allow_test_as_val,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    student = QCNet.load_from_checkpoint(checkpoint_path=args.student_ckpt, map_location=device).eval().to(device)
    teacher = QCNet.load_from_checkpoint(checkpoint_path=args.teacher_ckpt, map_location=device).eval().to(device)
    groups = defaultdict(make_bucket)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        student_pred = student(batch)
        teacher_pred = teacher(batch)
        eval_mask = batch["agent"]["category"] == 3
        if eval_mask.sum() == 0:
            continue

        valid_mask = batch["agent"]["predict_mask"][:, student.num_historical_steps:][eval_mask]
        gt_xy = batch["agent"]["target"][eval_mask, :, :2]
        sp = {k: v[eval_mask] if torch.is_tensor(v) and v.size(0) == eval_mask.size(0) else v
              for k, v in student_pred.items()}
        tp = {k: v[eval_mask] if torch.is_tensor(v) and v.size(0) == eval_mask.size(0) else v
              for k, v in teacher_pred.items()}
        student_prob = F.softmax(student._eval_pi_logits(student_pred, student_pred["pi"])[eval_mask], dim=-1)
        teacher_prob = F.softmax(teacher._eval_pi_logits(teacher_pred, teacher_pred["pi"])[eval_mask], dim=-1)

        student_propose_xy = sp["loc_propose_pos"][..., :2]
        student_refine_xy = sp["loc_refine_pos"][..., :2]
        teacher_propose_xy = tp["loc_propose_pos"][..., :2]
        teacher_refine_xy = tp["loc_refine_pos"][..., :2]

        student_propose_ade = mode_ade(student_propose_xy, gt_xy, valid_mask)
        student_propose_fde = mode_fde(student_propose_xy, gt_xy, valid_mask)
        student_refine_ade = mode_ade(student_refine_xy, gt_xy, valid_mask)
        student_refine_fde = mode_fde(student_refine_xy, gt_xy, valid_mask)
        teacher_propose_ade = mode_ade(teacher_propose_xy, gt_xy, valid_mask)
        teacher_propose_fde = mode_fde(teacher_propose_xy, gt_xy, valid_mask)
        teacher_refine_ade = mode_ade(teacher_refine_xy, gt_xy, valid_mask)
        teacher_refine_fde = mode_fde(teacher_refine_xy, gt_xy, valid_mask)

        agent_batch = batch["agent"]["batch"][eval_mask] if isinstance(batch, Batch) else torch.zeros(
            int(eval_mask.sum()), dtype=torch.long, device=device)
        cities = batch["city"] if isinstance(batch["city"], list) else [str(batch["city"])]
        for graph_id in agent_batch.unique(sorted=True):
            agent_sel = agent_batch == graph_id
            loc = str(cities[int(graph_id.detach().cpu())])
            for group_name in ("global", f"type:{scene_type(loc)}", f"location:{loc}"):
                bucket = groups[group_name]
                add_count(bucket, int(agent_sel.sum().cpu()))
                update_model(
                    bucket, "student",
                    student_propose_ade[agent_sel],
                    student_propose_fde[agent_sel],
                    student_refine_ade[agent_sel],
                    student_refine_fde[agent_sel],
                    student_prob[agent_sel],
                    student_refine_xy[agent_sel, :, -1],
                )
                update_model(
                    bucket, "teacher",
                    teacher_propose_ade[agent_sel],
                    teacher_propose_fde[agent_sel],
                    teacher_refine_ade[agent_sel],
                    teacher_refine_fde[agent_sel],
                    teacher_prob[agent_sel],
                    teacher_refine_xy[agent_sel, :, -1],
                )
                update_student_teacher_gap(
                    bucket,
                    student_refine_ade[agent_sel],
                    student_refine_fde[agent_sel],
                    teacher_refine_ade[agent_sel],
                    teacher_refine_fde[agent_sel],
                    student_refine_xy[agent_sel],
                    teacher_refine_xy[agent_sel],
                    valid_mask[agent_sel],
                )

        if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
            break

    return {group: finalize_bucket(bucket) for group, bucket in groups.items()}


def write_outputs(results: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "topossm_coverage_decomposition.json"
    csv_path = out_dir / "topossm_coverage_decomposition.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for group, metrics in sorted(results.items()):
        row = {"group": group}
        row.update(metrics)
        rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if "global" in results:
        print(json.dumps({"global": results["global"]}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--allow_test_as_val", action="store_true")
    args = parser.parse_args()
    write_outputs(evaluate(args), Path(args.out_dir))


if __name__ == "__main__":
    main()
