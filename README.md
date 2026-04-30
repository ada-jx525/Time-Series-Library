# BranchWorld Time-Series Forecasting

This repository is a cleaned research codebase for **BranchWorldModel**:
a memory-centric latent world model for long-term time-series forecasting.

The current focus is:

- frozen or lightweight time-series backbone
- branch-aware latent state encoding
- offline future branch prototype memory bank
- retrieval-conditioned multi-branch rollout
- MoE branch selector / expert head
- long-term forecasting experiments on ETT/custom benchmark data

## Core Files

```text
run.py                                      # experiment entry
exp/exp_long_term_forecasting.py            # train / validation / test loop
models/BranchWorldModel.py                  # BranchWorld model
layers/BranchWorld.py                       # backbone zoo, memory, rollout components
data_provider/data_loader.py                # ETT/custom long-term forecasting datasets
data_provider/data_factory.py               # data loader factory
research/                                  # experiment and ablation plans
scripts/long_term_forecast/ETT_script/      # BranchWorld ablation scripts
```

## Backbones

BranchWorld supports the following state encoder backbones:

```bash
--wm_backbone patch_transformer
--wm_backbone temporal_transformer
--wm_backbone inverted_transformer
--wm_backbone tcn
--wm_backbone mlp
```

The default research setting freezes the backbone:

```bash
--wm_freeze_backbone 1
```

This makes the method memory-centric: the trainable part is the branch memory
reasoning head, not a large forecasting backbone.

## Environment

Install PyTorch first according to your CUDA version, then install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Smoke Test

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id smoke_BranchWorld_ETTh1_96_96 \
  --model BranchWorldModel --data ETTh1 --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --n_heads 4 --e_layers 1 --d_layers 1 --d_ff 128 \
  --wm_latent_dim 64 --wm_branch_num 4 --wm_retrieve_k 4 \
  --wm_memory_size 512 --wm_backbone patch_transformer \
  --wm_freeze_backbone 1 --wm_head_type moe \
  --train_epochs 1 --batch_size 16 --itr 1
```

## Experiment Plans

See:

- `research/BranchWorld_experiment_execution_plan.md`
- `research/BranchWorld_ablation_plan.md`
