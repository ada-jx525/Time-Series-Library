# Time-Series-Library with World-Trajectory Memory

This repository keeps the classic long-term forecasting models and adds
`BranchWorldModel` as a memory-enhanced forecasting module.

`BranchWorldModel` follows the current task definition:

- encode the current window into latent state `z_i^0`
- store multi-horizon latent displacement trajectories `T_i`
- retrieve neighbors by current-state similarity only
- cluster retrieved trajectories into future trajectory prototypes `P_m`
- decode memory branch forecasts `y_m^mem`
- fuse `y_base` and memory branches with a learned reliability-aware gate

## Core Files

```text
run.py                            # experiment entry
exp/exp_long_term_forecasting.py  # train / validation / test loop
models/BranchWorldModel.py        # world-trajectory memory module
layers/BranchWorld.py             # pre-norm state encoder, decoder, clustering helpers
models/                           # classic baseline model implementations are retained
data_provider/                    # long-term forecasting datasets
```

## BranchWorldModel Options

```bash
--wm_backbone patch_transformer
--wm_branch_num 3
--wm_retrieve_k 64
--wm_memory_size 4096
--wm_memory_update_freq 1
--wm_horizons 1 4 8 16 32 64 96
--wm_freeze_base 1
--wm_mem_loss_type min
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
  --wm_latent_dim 64 --wm_branch_num 3 --wm_retrieve_k 32 \
  --wm_memory_size 512 --wm_backbone patch_transformer \
  --wm_memory_update_freq 1 \
  --train_epochs 1 --batch_size 16 --itr 1
```
