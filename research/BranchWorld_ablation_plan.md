# BranchWorldModel Ablation Plan

This file records the core ablations for the branch-aware latent world model.
The goal is to isolate whether the gain comes from branch-aware latent states,
offline future branch discovery, retrieval-conditioned rollout, and mixture
selection, rather than from a larger encoder alone.

## Main Variants

| ID | Variant | Command knobs | Question answered |
| --- | --- | --- | --- |
| A0 | Full BranchWorldModel | default BranchWorld flags | Does the complete method improve forecasting? |
| A1 | No memory | `--wm_use_memory 0` | Is offline branch memory useful beyond learned branch priors? |
| A2 | Raw retrieval, no branch discovery | `--wm_use_branch_discovery 0` | Are future branch prototypes better than raw exemplar retrieval? |
| A3 | Single branch | `--wm_branch_num 1 --wm_retrieve_k 1` | Does multi-branch rollout matter? |
| A4 | No learned gate | `--wm_use_gating 0` | Does branch selection improve over similarity-weighted mixing? |
| A5 | No auxiliary losses | `--wm_use_aux_losses 0` | Do transition-aware latent regularizers help? |
| A6 | No diversity regularization | `--wm_diversity_weight 0` | Does diversity preservation prevent branch collapse? |
| A7 | No oracle branch loss | `--wm_oracle_weight 0` | Does best-branch supervision make branches forecast-useful? |
| A8 | No latent consistency | `--wm_aux_weight 0` | Does latent rollout consistency improve dynamics modeling? |
| A9 | End-to-end backbone | `--wm_freeze_backbone 0` | Is memory reasoning still useful when the encoder is allowed to adapt? |
| A10 | Shared decoder instead of MoE | `--wm_head_type shared` | Does branch-specialized decoding improve over a shared decoder? |

## Recommended Tables

1. Main long-term forecasting table:
   - Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic, Exchange, ILI.
   - Horizons: 96, 192, 336, 720 where applicable.
   - Metrics: MSE and MAE.
   - Baselines: DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, TimeXer, and the strongest recent repo baselines available in this codebase.

2. Ablation table:
   - Use ETTh1, ETTm1, Weather, ECL as a compact but diverse set.
   - Report average MSE/MAE across four horizons.
   - Include A0-A8 variants.

3. Branch behavior analysis:
   - Average pairwise distance between branch predictions.
   - Gating entropy.
   - Oracle branch error vs. mixture error.
   - Retrieval similarity of selected branches.

4. Efficiency table:
   - Parameters.
   - Training time per epoch.
   - Inference time per batch.
   - Memory build time and memory size.

## Claims Each Ablation Should Support

- A0 vs. A1: retrieval provides a dynamics prior, not just extra parameters.
- A0 vs. A2: storing future branch prototypes is stronger than ordinary nearest-neighbor retrieval.
- A0 vs. A3: separate branch-conditioned rollouts reduce multimodal future averaging.
- A0 vs. A4: branch selector is necessary when multiple plausible futures are retrieved.
- A0 vs. A5-A8: the learned latent space is branch-aware and transition-aware, not a generic forecasting embedding.

## Minimal Run Example

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96_BranchWorld --model BranchWorldModel --data ETTh1 \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --wm_latent_dim 128 --wm_branch_num 4 --wm_retrieve_k 4
```

## Paper Positioning

Use the method section to emphasize that the default setting freezes the state
backbone and trains a memory reasoning head. The memory stores latent future
branch prototypes, not raw historical windows. Retrieval conditions the latent
transition law `z_{t+h}=f(z_{t+h-1}, r_k)`, so memory is a dynamics prior rather
than a decoder-side feature.
