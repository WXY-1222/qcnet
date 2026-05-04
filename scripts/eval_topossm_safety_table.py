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


def parse_model_specs(specs: Iterable[str]) -> List[Tuple[str, str]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be NAME=/path/to.ckpt, got {spec!r}")
        name, ckpt = spec.split("=", 1)
        out.append((name.strip(), ckpt.strip()))
    return out


def make_bucket():
    return defaultdict(float)


def add_count(bucket, n: int) -> None:
    bucket["n"] += int(n)


def add_mean(bucket, key: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    bucket[key + "_sum"] += float(values.detach().sum().cpu())
    bucket[key + "_n"] += int(values.numel())


def add_rate(bucket, key: str, numer: torch.Tensor, denom: torch.Tensor) -> None:
    bucket[key + "_num"] += float(numer.detach().sum().cpu())
    bucket[key + "_den"] += float(denom.detach().sum().cpu())


def finalize_bucket(bucket) -> Dict[str, float]:
    out = {"n": int(bucket.get("n", 0))}
    for key, val in bucket.items():
        if key.endswith("_sum"):
            base = key[:-4]
            den = bucket.get(base + "_n", 0.0)
            out[base] = float(val / den) if den else 0.0
        if key.endswith("_num"):
            base = key[:-4]
            den = bucket.get(base + "_den", 0.0)
            out[base] = float(val / den) if den else 0.0
    return out


def local_to_global(local_xy: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    cos, sin = theta.cos(), theta.sin()
    rot = local_xy.new_zeros((local_xy.size(0), 2, 2))
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = sin
    rot[:, 1, 0] = -sin
    rot[:, 1, 1] = cos
    return torch.einsum("nti,nij->ntj", local_xy, rot) + origin[:, :2].view(-1, 1, 2)


def nearest_map_stats(
    traj_global: torch.Tensor,
    gt_global: torch.Tensor,
    valid_mask: torch.Tensor,
    map_pos: torch.Tensor,
    edge_index: torch.Tensor,
    offroad_threshold: float,
) -> Dict[str, torch.Tensor]:
    n, t, _ = traj_global.shape
    flat_pred = traj_global.reshape(n * t, 2)
    flat_gt = gt_global.reshape(n * t, 2)
    if map_pos.numel() == 0:
        large = traj_global.new_full((n, t), 1e3)
        return {
            "corridor_dist": large,
            "offroad": large > offroad_threshold,
            "route_jump": valid_mask.new_zeros((n, max(t - 1, 0)), dtype=torch.bool),
            "route_jump_valid": valid_mask[:, 1:] & valid_mask[:, :-1],
            "route_node_match": valid_mask.new_zeros((n, t), dtype=torch.bool),
        }
    dist_pred = torch.cdist(flat_pred, map_pos).reshape(n, t, -1)
    dist_gt = torch.cdist(flat_gt, map_pos).reshape(n, t, -1)
    corridor_dist, pred_node = dist_pred.min(dim=-1)
    _, gt_node = dist_gt.min(dim=-1)

    if edge_index.numel() > 0:
        edges = set((int(a), int(b)) for a, b in edge_index.t().detach().cpu().tolist())
        edges |= set((b, a) for a, b in edges)
    else:
        edges = set()
    jumps = []
    for i in range(n):
        row = []
        for j in range(max(t - 1, 0)):
            a = int(pred_node[i, j].detach().cpu())
            b = int(pred_node[i, j + 1].detach().cpu())
            row.append(not (a == b or (a, b) in edges))
        jumps.append(row)
    route_jump = torch.tensor(jumps, device=traj_global.device, dtype=torch.bool) if t > 1 else valid_mask[:, :0]
    return {
        "corridor_dist": corridor_dist,
        "offroad": corridor_dist > offroad_threshold,
        "route_jump": route_jump,
        "route_jump_valid": valid_mask[:, 1:] & valid_mask[:, :-1],
        "route_node_match": pred_node == gt_node,
    }


def collision_stats(traj_global: torch.Tensor, valid_mask: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n, t, _ = traj_global.shape
    if n < 2:
        return traj_global.new_tensor(0.0), traj_global.new_tensor(0.0)
    coll_num = traj_global.new_tensor(0.0)
    coll_den = traj_global.new_tensor(0.0)
    for step in range(t):
        valid = valid_mask[:, step]
        idx = valid.nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        d = torch.cdist(traj_global[idx, step], traj_global[idx, step])
        tri = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
        coll_num = coll_num + (d[tri] < threshold).to(dtype=traj_global.dtype).sum()
        coll_den = coll_den + tri.sum().to(dtype=traj_global.dtype)
    return coll_num, coll_den


def update_group(
    bucket,
    ade_best,
    fde_best,
    mr_best,
    ade_top1,
    fde_top1,
    brier_fde_best,
    top1_prob,
    corridor_best,
    corridor_top1,
    offroad_best,
    offroad_top1,
    route_jump_best,
    route_jump_top1,
    route_jump_valid,
    route_match_best,
    route_match_top1,
    valid_mask,
) -> None:
    add_count(bucket, ade_best.numel())
    add_mean(bucket, "minADE", ade_best)
    add_mean(bucket, "minFDE", fde_best)
    add_mean(bucket, "MR", mr_best)
    add_mean(bucket, "top1_ADE", ade_top1)
    add_mean(bucket, "top1_FDE", fde_top1)
    add_mean(bucket, "brier_FDE_best", brier_fde_best)
    add_mean(bucket, "top1_prob", top1_prob)
    add_rate(bucket, "corridor_dist_best", corridor_best * valid_mask, valid_mask)
    add_rate(bucket, "corridor_dist_top1", corridor_top1 * valid_mask, valid_mask)
    add_rate(bucket, "offroad_best", offroad_best & valid_mask, valid_mask)
    add_rate(bucket, "offroad_top1", offroad_top1 & valid_mask, valid_mask)
    add_rate(bucket, "route_jump_best", route_jump_best & route_jump_valid, route_jump_valid)
    add_rate(bucket, "route_jump_top1", route_jump_top1 & route_jump_valid, route_jump_valid)
    add_rate(bucket, "route_node_match_best", route_match_best & valid_mask, valid_mask)
    add_rate(bucket, "route_node_match_top1", route_match_top1 & valid_mask, valid_mask)


@torch.no_grad()
def evaluate_model(args, name: str, ckpt_path: str, loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, float]]:
    model = QCNet.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=device)
    model.eval().to(device)
    groups = defaultdict(make_bucket)

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        eval_mask = batch["agent"]["category"] == 3
        if eval_mask.sum() == 0:
            continue
        reg_mask = batch["agent"]["predict_mask"][:, model.num_historical_steps:][eval_mask]
        local_gt = batch["agent"]["target"][eval_mask, :, :2]
        local_pred = pred["loc_refine_pos"][eval_mask, :, :, :2]
        pi_logits = model._eval_pi_logits(pred, pred["pi"]) if hasattr(model, "_eval_pi_logits") else pred["pi"]
        prob = F.softmax(pi_logits[eval_mask], dim=-1)
        top_idx = prob.argmax(dim=-1)
        n, k, t, _ = local_pred.shape
        valid_counts = reg_mask.sum(dim=-1).clamp(min=1)
        last_idx = (reg_mask * torch.arange(1, t + 1, device=device)).argmax(dim=-1)

        fde_modes = torch.norm(
            local_pred[torch.arange(n, device=device), :, last_idx] -
            local_gt[torch.arange(n, device=device), last_idx].unsqueeze(1),
            dim=-1,
        )
        best_idx = fde_modes.argmin(dim=-1)
        best_prob = prob[torch.arange(n, device=device), best_idx]
        brier_fde_best = (1.0 - best_prob).pow(2)
        top1_prob = prob[torch.arange(n, device=device), top_idx]
        pred_best = local_pred[torch.arange(n, device=device), best_idx]
        pred_top1 = local_pred[torch.arange(n, device=device), top_idx]
        ade_best = (torch.norm(pred_best - local_gt, dim=-1) * reg_mask).sum(dim=-1) / valid_counts
        fde_best = fde_modes[torch.arange(n, device=device), best_idx]
        mr_best = (fde_best > args.miss_threshold).to(dtype=ade_best.dtype)
        ade_top1 = (torch.norm(pred_top1 - local_gt, dim=-1) * reg_mask).sum(dim=-1) / valid_counts
        fde_top1 = torch.norm(
            pred_top1[torch.arange(n, device=device), last_idx] -
            local_gt[torch.arange(n, device=device), last_idx],
            dim=-1,
        )

        origin = batch["agent"]["position"][eval_mask, model.num_historical_steps - 1]
        theta = batch["agent"]["heading"][eval_mask, model.num_historical_steps - 1]
        pred_best_global = local_to_global(pred_best, origin, theta)
        pred_top1_global = local_to_global(pred_top1, origin, theta)
        gt_global = batch["agent"]["position"][eval_mask, model.num_historical_steps:, :2]

        agent_batch = batch["agent"]["batch"][eval_mask] if isinstance(batch, Batch) else torch.zeros(n, dtype=torch.long, device=device)
        map_batch = batch["map_polygon"]["batch"] if isinstance(batch, Batch) else torch.zeros(
            batch["map_polygon"]["position"].size(0), dtype=torch.long, device=device)
        cities = batch["city"] if isinstance(batch["city"], list) else [str(batch["city"])]
        for graph_id in agent_batch.unique(sorted=True):
            agent_sel = agent_batch == graph_id
            graph_int = int(graph_id.detach().cpu())
            loc = str(cities[graph_int])
            typ = scene_type(loc)
            map_sel = map_batch == graph_id
            map_pos = batch["map_polygon"]["position"][map_sel, :2]
            global_map_idx = map_sel.nonzero(as_tuple=False).flatten()
            global_to_local = {int(g.detach().cpu()): i for i, g in enumerate(global_map_idx)}
            edge = batch["map_polygon", "to", "map_polygon"]["edge_index"]
            keep = map_sel[edge[0]] & map_sel[edge[1]]
            edge_local = []
            for a, b in edge[:, keep].t().detach().cpu().tolist():
                edge_local.append([global_to_local[int(a)], global_to_local[int(b)]])
            edge_local = torch.tensor(edge_local, device=device, dtype=torch.long).t() if edge_local else torch.zeros((2, 0), device=device, dtype=torch.long)

            best_stats = nearest_map_stats(
                pred_best_global[agent_sel], gt_global[agent_sel], reg_mask[agent_sel],
                map_pos, edge_local, args.offroad_threshold)
            top1_stats = nearest_map_stats(
                pred_top1_global[agent_sel], gt_global[agent_sel], reg_mask[agent_sel],
                map_pos, edge_local, args.offroad_threshold)
            coll_best_num, coll_best_den = collision_stats(pred_best_global[agent_sel], reg_mask[agent_sel], args.collision_threshold)
            coll_top1_num, coll_top1_den = collision_stats(pred_top1_global[agent_sel], reg_mask[agent_sel], args.collision_threshold)

            for group_name in ["global", f"location:{loc}", f"type:{typ}"]:
                bucket = groups[group_name]
                update_group(
                    bucket,
                    ade_best[agent_sel], fde_best[agent_sel], mr_best[agent_sel],
                    ade_top1[agent_sel], fde_top1[agent_sel],
                    brier_fde_best[agent_sel], top1_prob[agent_sel],
                    best_stats["corridor_dist"], top1_stats["corridor_dist"],
                    best_stats["offroad"], top1_stats["offroad"],
                    best_stats["route_jump"], top1_stats["route_jump"],
                    best_stats["route_jump_valid"],
                    best_stats["route_node_match"], top1_stats["route_node_match"],
                    reg_mask[agent_sel],
                )
                bucket["collision_best_num"] += float(coll_best_num.cpu())
                bucket["collision_best_den"] += float(coll_best_den.cpu())
                bucket["collision_top1_num"] += float(coll_top1_num.cpu())
                bucket["collision_top1_den"] += float(coll_top1_den.cpu())

    return {group: finalize_bucket(bucket) for group, bucket in groups.items()}


def write_outputs(results, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "topossm_safety_eval.json"
    csv_path = out_dir / "topossm_safety_eval.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for model_name, groups in results.items():
        for group_name, metrics in groups.items():
            row = {"model": model_name, "group": group_name}
            row.update(metrics)
            rows.append(row)
    keys = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--offroad_threshold", type=float, default=2.0)
    parser.add_argument("--collision_threshold", type=float, default=2.0)
    parser.add_argument("--miss_threshold", type=float, default=2.0)
    parser.add_argument("--model", action="append", required=True, help="NAME=/path/to/checkpoint.ckpt")
    args = parser.parse_args()

    dataset = InteractionDIGIRDataset(
        data_path=args.data,
        split="val",
        transform=TargetBuilder(8, 12),
        num_historical_steps=8,
        num_future_steps=12,
        max_samples=args.max_samples,
        use_kg=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}
    for name, ckpt in parse_model_specs(args.model):
        print(f"evaluating {name}: {ckpt}")
        results[name] = evaluate_model(args, name, ckpt, loader, device)
    write_outputs(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
