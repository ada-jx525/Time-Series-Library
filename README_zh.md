# Time-Series-Library with World-Trajectory Memory

本仓库保留 classic long-term forecasting 模型，并新增 `BranchWorldModel`
作为 memory-enhanced forecasting module。

当前 `BranchWorldModel` 按任务书实现：

- 编码当前窗口得到 latent state `z_i^0`
- memory 存 multi-horizon latent displacement trajectory `T_i`
- 检索只使用当前状态相似性
- 离线从训练集 `T_i` 学习全局 latent dynamics prototype bank
- 在线根据检索邻居从 bank 中选择 query-relevant prototypes
- 由轻量 decoder 生成 memory branch forecast `y_m^mem`
- 用 reliability-aware gate 融合 `y_base` 和 memory branches

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
--wm_branch_num 3
--wm_retrieve_k 64
--wm_memory_size 4096
--wm_global_proto_num 32
--wm_proto_mode offline
--wm_memory_update_freq 1
--wm_horizons 1 4 8 16 32 64 96
--wm_freeze_base 1
--wm_mem_loss_type min
```

如果不传 `--wm_horizons`，默认使用 `{H/4, H/2, 3H/4, H}`。
