# PaperA h8_f12_s5 Clean No-Distill Best Metrics

Date: 2026-05-07

This note records the best clean no-distillation PaperA / TopoSSM full-decoder results found in the parameter sweep from the 0.0531 lineage. It is intended to keep the historically best metric, checkpoint path, and key conclusion in git without committing heavy checkpoints or run directories.

## Current Best

- Best candidate: `P07`
- Best `val_minADE`: `0.05270220711827278`
- Best checkpoint: `/data/sdb/bitwxy/qcnet_data/paperA_param_sweep_h8_f12_s5_k6_4gpu_20260507_paperA_param_sweep_from0531_v1/P07_parent_lr1p5e-5_mlr5e-6_wd1e-4_s0p16_c0p00_fz0/lightning_logs/version_0/checkpoints/epoch=2-step=483.ckpt`
- Init checkpoint: `/data/sdb/bitwxy/qcnet_data/paperA_seed23_lowlr_cont_h8_f12_s5_k6_4gpu_20260507_paperA_seed23_lowlr_cont/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt`
- Config: `lr=1.5e-5`, `mamba_lr=5e-6`, `weight_decay=1e-4`, `topo_mode_endpoint_scale=0.16`, `topo_corridor_loss_weight=0.00`, `freeze_encoder=0`

## Interpretation

`P07` was not obtained by directly fine-tuning from the previous `0.0531` best checkpoint. It replayed from the parent checkpoint that preceded the `0.0531` run, with `topo_corridor_loss_weight` set to `0.00`.

The key empirical signal is that keeping the topology/corridor structure in the decoder while removing the explicit corridor auxiliary loss improved ADE. This suggests the corridor auxiliary supervision was constraining or distracting the best ADE mode, while the structural TopoSSM path can still use corridor information through the trajectory objective.

## Nearest Comparisons

- `B02`: `0.052846137434244156`, direct continuation from the previous `0.0531` best. Close, but weaker than `P07`.
- `P06`: `0.053115230053663254`, same parent replay with `topo_corridor_loss_weight=0.01`. Close to the previous best, but weaker than `P07`.
- Previous clean best: about `0.053098794`, from `paperA_from054_finetune_v1`.

The full summary table is in `results/paperA_h8f12s5_clean_best_metrics_20260507.csv`.
