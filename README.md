# Time-Series-Library with World-Trajectory Memory

This repository keeps the classic long-term forecasting models and adds
`BranchWorldModel` as a memory-enhanced forecasting module.

`BranchWorldModel` follows the current task definition:

- encode the current window into latent state `z_i^0`
- store multi-horizon latent displacement trajectories `T_i`
- learn an offline global latent dynamics prototype bank from training `T_i`
- store one mean base-residual correction per trajectory prototype
- use online soft attention from the current key to prototype keys
- return `y_base + lambda * weighted_residual`

The memory bank is built from the training split only. Validation and test
windows never contribute future values to the offline prototypes.

## Core Files

```text
run.py                            # experiment entry
exp/exp_long_term_forecasting.py  # train / validation / test loop
models/BranchWorldModel.py        # world-trajectory memory module
layers/BranchWorld.py             # pre-norm state encoder and clustering helpers
models/                           # classic baseline model implementations are retained
data_provider/                    # long-term forecasting datasets
```

## BranchWorldModel Options

```bash
--wm_backbone patch_transformer
--wm_memory_size 4096
--wm_global_proto_num 32
--wm_memory_update_freq 1
--wm_horizons 1 4 8 16 32 64 96
--wm_freeze_base 1
--wm_freeze_encoder 1
--wm_attention_mode dot
--wm_correction_lambda 0.1
--wm_predictor_loss ce
--wm_predictor_checkpoint ""
--wm_predictor_ce_weight 1.0
--wm_predictor_ce_only 1
--wm_predictor_entropy_weight 0.0
--wm_branch_temperature 1.0
```

If `--wm_horizons` is omitted, the model uses `{H/4, H/2, 3H/4, H}`.

## Smoke Test

Install PyTorch for your CUDA version first, then:

```bash
pip install -r requirements.txt
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id smoke_wtm_ETTh1_96_96 \
  --model BranchWorldModel --data ETTh1 --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --n_heads 4 --e_layers 1 --d_layers 1 --d_ff 128 \
  --wm_latent_dim 64 \
  --wm_memory_size 512 --wm_global_proto_num 16 --wm_backbone patch_transformer \
  --wm_memory_update_freq 1 --wm_correction_lambda 0.1 \
  --train_epochs 1 --batch_size 16 --itr 1
```

## Clustering Diagnostic

Use this before training gates or adapters. It builds latent trajectory items
from the training split only, runs k-means for several `K`, and reports:

- k-means inertia for choosing `K`
- cluster-weighted residual variance vs global residual variance
- linear-probe accuracy from current-window key to trajectory cluster label
- soft-attention entropy from current key to prototype keys

```bash
python -u scripts/diagnostics/branchworld_clustering_quality.py \
  @configs/benchmark/etth1_common.args \
  @configs/benchmark/branchworld_common.args \
  --no_use_gpu --num_workers 0 --batch_size 128 \
  --diag_k_values 4 8 16 32 64 \
  --diag_probe_epochs 300 \
  --diag_output_csv results/branchworld_cluster_predictability_etth1.csv
```
