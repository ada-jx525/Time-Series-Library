# BranchWorldModel Experiment Execution Plan

This plan follows the experimental style of recent top-tier time-series papers:
PatchTST (ICLR 2023), TimesNet (ICLR 2023), iTransformer (ICLR 2024),
TimeMixer (ICLR 2024), and TimeXer (NeurIPS 2024).

## 0. Goal

The paper should not be positioned as "another forecasting backbone." The core
claim is:

> A forecast can be improved by learning a branch-aware latent world state,
> discovering offline future-evolution branches, and using retrieved branch
> prototypes as dynamics priors for multi-branch latent rollout.

The experiments must prove three things:

1. Accuracy: BranchWorldModel is competitive or SOTA on standard forecasting benchmarks.
2. Mechanism: gains come from branch-aware memory and rollout, not from parameter count.
3. Behavior: retrieved branches represent different plausible future evolutions.

## 1. Environment and Data

### 1.1 Environment

Create a clean environment before large experiments:

```bash
conda create -n branchworld python=3.11
conda activate branchworld
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If running only local CSV long-term forecasting, the code now tolerates missing
optional dependencies such as `sktime`, `patool`, and `huggingface_hub`. For full
benchmark coverage, install all requirements.

### 1.2 Dataset Layout

Place datasets under:

```text
dataset/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
  exchange_rate/exchange_rate.csv
  illness/national_illness.csv
```

## 2. Stage-By-Stage Execution

### Backbone Choice

BranchWorldModel now has an explicit state-encoder backbone switch:

```text
--wm_backbone patch_transformer       # default, recommended first choice
--wm_backbone temporal_transformer    # vanilla temporal Transformer
--wm_backbone inverted_transformer    # iTransformer-style variable tokens
--wm_backbone tcn                     # convolutional temporal encoder
--wm_backbone mlp                     # simple control baseline
```

Default recommendation:

- Main method: `frozen patch_transformer + MoE memory head`.
- Backbone ablation: compare all five backbones on ETTh1, ETTm1, Weather.
- End-to-end backbone training is an upper-bound variant. Use
  `--wm_freeze_backbone 0` as an ablation or enhanced setting.

Reason:

- The paper is memory-centric. Freezing the backbone reduces compute and makes
  the central claim cleaner: improvements should come from branch prototype
  memory, retrieval-conditioned rollout, and MoE selection rather than from a
  larger trainable encoder.
- The trainable components remain the projection heads, future trajectory
  encoder, memory-conditioned rollout, MoE router, and expert heads.

### Stage 1: Smoke Test

Purpose: verify the full code path on one small run before spending GPU time.

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
  --wm_memory_size 512 --wm_backbone patch_transformer --wm_freeze_backbone 1 --wm_head_type moe \
  --train_epochs 1 --batch_size 16 --itr 1
```

Pass criterion:

- Offline branch memory is built.
- Training and testing finish.
- `results/.../metrics.npy`, `pred.npy`, and `true.npy` are saved.

### Stage 2: Main Long-Term Forecasting Table

Use the standard long-term forecasting protocol:

- Non-ILI horizons: `pred_len in {96, 192, 336, 720}`.
- ILI horizons: `pred_len in {24, 36, 48, 60}`.
- Default look-back: `seq_len=96`, `label_len=48`.
- Metrics: MSE and MAE.
- Repeat: start with `itr=1`; for final paper, run 3 seeds if compute allows.

Datasets, in recommended order:

1. ETTh1, ETTh2, ETTm1, ETTm2
2. Weather, Exchange
3. ECL, Traffic
4. ILI

Baselines to report:

- Classical strong simple baseline: DLinear.
- Transformer-family: Autoformer, FEDformer, PatchTST, iTransformer.
- General/SOTA recent baselines: TimesNet, TimeMixer, TimeXer if using exogenous setup.
- Repository recent additions if relevant: WPMixer, TimeFilter, MultiPatchFormer.

### Stage 3: Look-Back Window Study

Recent papers often show that conclusions can change when `seq_len` changes.
Run at least:

```text
seq_len in {96, 192, 336, 512}
pred_len in {96, 192, 336, 720}
```

Datasets:

- ETTh1
- ETTm1
- Weather
- ECL

Claim to test:

- BranchWorld should benefit from longer context because retrieval-worthy latent
  states become more informative.

### Stage 4: Ablation Study

Use compact but representative datasets:

- ETTh1
- ETTm1
- Weather
- ECL

Variants:

| Variant | Flags |
| --- | --- |
| Full | default |
| No memory | `--wm_use_memory 0` |
| Raw retrieval | `--wm_use_branch_discovery 0` |
| Single branch | `--wm_branch_num 1 --wm_retrieve_k 1` |
| No gating | `--wm_use_gating 0` |
| No aux losses | `--wm_use_aux_losses 0` |
| No diversity | `--wm_diversity_weight 0` |
| No oracle branch loss | `--wm_oracle_weight 0` |
| No latent consistency | `--wm_aux_weight 0` |
| Shared decoder head | `--wm_head_type shared` |
| End-to-end state backbone | `--wm_freeze_backbone 0` |
| Backbone swap | `--wm_backbone temporal_transformer / inverted_transformer / tcn / mlp` |

Report:

- Average MSE/MAE over four horizons.
- Relative degradation vs. Full.

Expected interpretation:

- Full vs. No memory: proves retrieval dynamics prior matters.
- Full vs. Raw retrieval: proves branch prototypes beat ordinary exemplar retrieval.
- Full vs. Single branch: proves multi-branch rollout avoids future averaging.
- Full vs. No gating: proves branch selector is necessary.
- Full vs. No aux: proves latent state is transition-aware, not generic.

### Stage 5: Branch Behavior Analysis

This is important for a CCF-A-level story. Accuracy alone is not enough.

Run analysis on ETTh1, Weather, and ECL:

1. Branch diversity:
   - Average pairwise distance between branch predictions.
   - Compare Full vs. No diversity and Single branch.

2. Gating entropy:
   - Low entropy on deterministic segments.
   - Higher entropy near regime shifts or ambiguous futures.

3. Oracle branch gap:
   - Compute best branch error and mixture error.
   - If best branch is much better than mixture, improve gating.

4. Retrieved branch visualization:
   - Plot current window, ground truth future, top-K branch predictions.
   - Show branch prototypes correspond to different future trajectories.

5. Retrieval sanity:
   - Compare retrieved neighbors in raw input space vs. latent state space.
   - Show latent retrieval clusters future dynamics more cleanly.

### Stage 6: Efficiency Analysis

Report:

- Parameters.
- Training time per epoch.
- Inference time per batch.
- Memory build time.
- Memory size sensitivity: `wm_memory_size in {512, 1024, 2048, 4096}`.
- Branch count sensitivity: `wm_branch_num in {1, 2, 4, 8}`.

The method has extra memory and rollout cost, so the paper must explicitly show
that the gain is worth the overhead.

### Stage 7: Hyperparameter Search

Start with:

```text
d_model:       128, 256
wm_latent_dim: 64, 128, 256
wm_branch_num: 2, 4, 8
wm_retrieve_k: 2, 4, 8
wm_memory_size: 1024, 2048, 4096
learning_rate: 1e-4, 5e-4, 1e-3
```

Recommended tuning order:

1. Tune `d_model`, `wm_latent_dim` on ETTh1 horizon 96 and 336.
2. Tune `wm_branch_num`, `wm_retrieve_k`.
3. Tune auxiliary weights.
4. Transfer the best setting to other datasets with minimal changes.

## 3. Paper Figures

Required figures:

1. Main framework figure:
   - Offline branch memory construction.
   - Online retrieval-conditioned multi-branch rollout.
   - Branch selector and mixture output.

2. Ablation bar plot:
   - Full vs. variants.

3. Branch visualization:
   - top-K branch rollouts and final mixture.

4. Memory analysis:
   - t-SNE/UMAP of latent states colored by future branch cluster.

5. Efficiency/accuracy tradeoff:
   - MSE vs. memory size or branch count.

## 4. Decision Gates

Do not continue to a full CCF-A submission unless these gates pass:

1. Full beats strong baselines on at least several major datasets/horizons.
2. Full is consistently better than No memory, Raw retrieval, and Single branch.
3. Branch visualizations show genuinely different future modes.
4. Efficiency overhead is acceptable or clearly justified.
5. Results are reproducible across at least 3 seeds for the main claims.

## 5. Risk and Fixes

| Risk | Symptom | Fix |
| --- | --- | --- |
| Branch collapse | branch predictions overlap | increase diversity weight, branch count, or oracle loss |
| Bad gating | oracle branch much better than mixture | add horizon embedding or stronger gate MLP |
| Retrieval noise | memory hurts vs. no memory | improve state encoder, increase memory size, normalize future codes |
| Slow memory build | large Traffic/ECL overhead | precompute memory cache, reduce memory_size, approximate nearest neighbor |
| Weak SOTA | full improves ablation but not baselines | use stronger backbone encoder or hybrid PatchTST/iTransformer encoder |
