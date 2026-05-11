# Time-Series-Library with World-Trajectory Memory

本仓库保留 classic long-term forecasting 模型，并新增 `BranchWorldModel`
作为 memory-enhanced forecasting module。

当前 `BranchWorldModel` 按任务书实现：

- 编码当前窗口得到 latent state `z_i^0`
- memory 存 multi-horizon latent displacement trajectory `T_i`
- 离线从训练集 `T_i` 学习全局 latent dynamics prototype bank
- 每个 trajectory prototype 保存一个平均 base-residual correction
- 在线用当前 key 对 prototype key 做 soft attention
- 输出 `y_base + lambda * weighted_residual`

memory bank 只由训练集构造。验证集和测试集的 future value 不会进入离线 prototype。

核心文件：

```text
run.py
exp/exp_long_term_forecasting.py
models/BranchWorldModel.py
layers/BranchWorld.py
models/                           # classic baseline 模型保留
data_provider/
```

常用参数：

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

如果不传 `--wm_horizons`，默认使用 `{H/4, H/2, 3H/4, H}`。

聚类质量诊断：

```bash
python -u scripts/diagnostics/branchworld_clustering_quality.py \
  @configs/benchmark/etth1_common.args \
  @configs/benchmark/branchworld_common.args \
  --no_use_gpu --num_workers 0 --batch_size 128 \
  --diag_k_values 4 8 16 32 64 \
  --diag_probe_epochs 300 \
  --diag_output_csv results/branchworld_cluster_predictability_etth1.csv
```

这个脚本只用训练集构造 `T_i`，输出 k-means inertia 曲线和
cluster 内 residual 方差 / 全局 residual 方差，同时输出 current-window
key 到 trajectory cluster label 的 linear probe accuracy，以及 current key
到 prototype key 的 soft-attention entropy。
