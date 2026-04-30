# BranchWorld 时间序列预测代码库

这是为当前课题清理后的研究代码库，核心目标是：

- 冻结或轻量时间序列 backbone
- 构建 branch-aware latent state
- 离线发现未来演化分支原型 memory bank
- 检索条件化 latent rollout
- MoE branch selector / expert head
- 长期时间序列预测实验

核心代码：

```text
run.py
exp/exp_long_term_forecasting.py
models/BranchWorldModel.py
layers/BranchWorld.py
data_provider/data_loader.py
research/
scripts/long_term_forecast/ETT_script/
```

默认主方法使用：

```bash
--wm_backbone patch_transformer
--wm_freeze_backbone 1
--wm_head_type moe
```

安装依赖：

```bash
pip install -r requirements.txt
```

完整实验方案见：

```text
research/BranchWorld_experiment_execution_plan.md
research/BranchWorld_ablation_plan.md
```
