#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    bucket[key + "_sum"] += float(values.sum().cpu())
    bucket[key + "_n"] += int(values.numel())


def add_rate(bucket, key: str, numer: torch.Tensor, denom: torch.Tensor | None = None) -> None:
    numer = numer.detach()
    if denom is None:
        denom = torch.ones_like(numer, dtype=torch.bool)
    denom = denom.detach().bool()
    bucket[key + "_num"] += float(numer[denom].float().sum().cpu())
    bucket[key + "_den"] += int(denom.sum().cpu())


def finalize_bucket(bucket) -> Dict[str, float]:
    out = {"n": int(bucket.get("n", 0))}
    for key, val in bucket.items():
        if key.endswith("_sum"):
            base = key[:-4]
            den = bucket.get(base + "_n", 0.0)
            out[base] = float(val / den) if den else 0.0
        elif key.endswith("_num"):
            base = key[:-4]
            den = bucket.get(base + "_den", 0.0)
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


def pairwise_traj_ade(
        student_xy: torch.Tensor,
        teacher_xy: torch.Tensor,
        valid_mask: torch.Tensor) -> torch.Tensor:
    dist = torch.norm(
        student_xy[:, :, None] - teacher_xy[:, None],
        dim=-1,
    )
    valid = valid_mask[:, None, None].to(dtype=dist.dtype)
    return (dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)


def topk_contains_best(prob: torch.Tensor, best_mode: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, prob.size(-1))
    topk = prob.topk(k, dim=-1).indices
    return (topk == best_mode.unsqueeze(-1)).any(dim=-1)


def update_model_metrics(
        bucket,
        prefix: str,
        propose_ade: torch.Tensor,
        propose_fde: torch.Tensor,
        refine_ade: torch.Tensor,
        refine_fde: torch.Tensor,
        prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = refine_ade.size(0)
    idx = torch.arange(n, device=refine_ade.device)
    propose_best = propose_ade.argmin(dim=-1)
    refine_best = refine_ade.argmin(dim=-1)
    score_top1 = prob.argmax(dim=-1)

    add_mean(bucket, f"{prefix}_propose_oracle_ADE", propose_ade[idx, propose_best])
    add_mean(bucket, f"{prefix}_propose_oracle_FDE", propose_fde[idx, propose_best])
    add_mean(bucket, f"{prefix}_refine_oracle_ADE", refine_ade[idx, refine_best])
    add_mean(bucket, f"{prefix}_refine_oracle_FDE", refine_fde[idx, refine_best])
    add_mean(bucket, f"{prefix}_refine_top1_ADE", refine_ade[idx, score_top1])
    add_mean(bucket, f"{prefix}_refine_top1_FDE", refine_fde[idx, score_top1])
    add_mean(bucket, f"{prefix}_oracle_gap_ADE", refine_ade[idx, score_top1] - refine_ade[idx, refine_best])
    add_mean(bucket, f"{prefix}_oracle_best_prob", prob[idx, refine_best])
    add_mean(bucket, f"{prefix}_top1_prob", prob[idx, score_top1])
    add_rate(bucket, f"{prefix}_top1_is_oracle", score_top1 == refine_best)
    add_rate(bucket, f"{prefix}_top2_contains_oracle", topk_contains_best(prob, refine_best, 2))
    add_rate(bucket, f"{prefix}_top3_contains_oracle", topk_contains_best(prob, refine_best, 3))
    add_mean(bucket, f"{prefix}_propose_to_refine_oracle_delta",
             propose_ade[idx, propose_best] - refine_ade[idx, refine_best])
    return refine_best, score_top1


def update_pair_metrics(
        bucket,
        student_pred: Dict[str, torch.Tensor],
        teacher_pred: Dict[str, torch.Tensor],
        student_prob: torch.Tensor,
        teacher_prob: torch.Tensor,
        valid_mask: torch.Tensor) -> None:
    idx = torch.arange(student_prob.size(0), device=student_prob.device)
    teacher_top1 = teacher_prob.argmax(dim=-1)
    student_top1 = student_prob.argmax(dim=-1)

    for key, label in (("loc_propose_pos", "propose"), ("loc_refine_pos", "refine")):
        student_xy = student_pred[key][..., :2]
        teacher_xy = teacher_pred[key][..., :2]
        pair_ade = pairwise_traj_ade(student_xy, teacher_xy, valid_mask)
        min_student_to_teacher = pair_ade.min(dim=2).values
        min_teacher_to_student, assigned_student = pair_ade.min(dim=1)
        teacher_top1_student = assigned_student.gather(1, teacher_top1.view(-1, 1)).squeeze(1)

        add_mean(bucket, f"student_{label}_mean_minADE_to_teacher_modes", min_student_to_teacher.mean(dim=-1))
        add_mean(bucket, f"teacher_{label}_mean_minADE_to_student_modes", min_teacher_to_student.mean(dim=-1))
        add_mean(bucket, f"teacher_{label}_top1_ADE_to_nearest_student",
                 min_teacher_to_student[idx, teacher_top1])
        add_mean(bucket, f"student_{label}_top1_ADE_to_nearest_teacher",
                 min_student_to_teacher[idx, student_top1])
        add_rate(bucket, f"{label}_student_top1_matches_teacher_top1_nearest", student_top1 == teacher_top1_student)


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
        student_logits = student._eval_pi_logits(student_pred, student_pred["pi"])[eval_mask]
        teacher_logits = teacher._eval_pi_logits(teacher_pred, teacher_pred["pi"])[eval_mask]
        student_prob = F.softmax(student_logits, dim=-1)
        teacher_prob = F.softmax(teacher_logits, dim=-1)

        student_propose_ade = mode_ade(sp["loc_propose_pos"][..., :2], gt_xy, valid_mask)
        student_propose_fde = mode_fde(sp["loc_propose_pos"][..., :2], gt_xy, valid_mask)
        student_refine_ade = mode_ade(sp["loc_refine_pos"][..., :2], gt_xy, valid_mask)
        student_refine_fde = mode_fde(sp["loc_refine_pos"][..., :2], gt_xy, valid_mask)
        teacher_propose_ade = mode_ade(tp["loc_propose_pos"][..., :2], gt_xy, valid_mask)
        teacher_propose_fde = mode_fde(tp["loc_propose_pos"][..., :2], gt_xy, valid_mask)
        teacher_refine_ade = mode_ade(tp["loc_refine_pos"][..., :2], gt_xy, valid_mask)
        teacher_refine_fde = mode_fde(tp["loc_refine_pos"][..., :2], gt_xy, valid_mask)

        agent_batch = batch["agent"]["batch"][eval_mask] if isinstance(batch, Batch) else torch.zeros(
            int(eval_mask.sum()), dtype=torch.long, device=device)
        cities = batch["city"] if isinstance(batch["city"], list) else [str(batch["city"])]
        for graph_id in agent_batch.unique(sorted=True):
            agent_sel = agent_batch == graph_id
            graph_int = int(graph_id.detach().cpu())
            loc = str(cities[graph_int])
            typ = scene_type(loc)
            for group_name in ("global", f"location:{loc}", f"type:{typ}"):
                bucket = groups[group_name]
                add_count(bucket, int(agent_sel.sum().cpu()))
                update_model_metrics(
                    bucket, "student",
                    student_propose_ade[agent_sel],
                    student_propose_fde[agent_sel],
                    student_refine_ade[agent_sel],
                    student_refine_fde[agent_sel],
                    student_prob[agent_sel],
                )
                update_model_metrics(
                    bucket, "teacher",
                    teacher_propose_ade[agent_sel],
                    teacher_propose_fde[agent_sel],
                    teacher_refine_ade[agent_sel],
                    teacher_refine_fde[agent_sel],
                    teacher_prob[agent_sel],
                )
                update_pair_metrics(
                    bucket,
                    {k: v[agent_sel] if torch.is_tensor(v) and v.size(0) == agent_sel.size(0) else v
                     for k, v in sp.items()},
                    {k: v[agent_sel] if torch.is_tensor(v) and v.size(0) == agent_sel.size(0) else v
                     for k, v in tp.items()},
                    student_prob[agent_sel],
                    teacher_prob[agent_sel],
                    valid_mask[agent_sel],
                )

        if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
            break

    return {group: finalize_bucket(bucket) for group, bucket in groups.items()}


def write_outputs(results: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "topossm_proposal_diagnostics.json"
    csv_path = out_dir / "topossm_proposal_diagnostics.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for group, metrics in sorted(results.items()):
        row = {"group": group}
        row.update(metrics)
        rows.append(row)
    keys = sorted({k for row in rows for k in row.keys()})
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--allow_test_as_val", action="store_true")
    args = parser.parse_args()
    results = evaluate(args)
    write_outputs(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
