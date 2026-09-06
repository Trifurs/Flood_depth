# PA-HydroKAN-v13.2 科学优化报告

## 结论

本轮优化完成了“物理边特征 + KAN 图消息”的新架构、受控消融、最终长训和回归验证。最终 raw 权重在验证集的主指标 `pixel_micro_mae=0.48025 m`，优于同架构短训 KAN-A（0.49402 m），但略差于既有校正 v13 基线（0.47345 m，约 +1.44%）。因此当前生产建议继续使用旧校正 v13；v13.2 raw 保留为研究候选，不宣称已取得总体精度提升。全程未使用 test split。

## 数据与可复现性

- 数据集：`subset150`；固定配置 `hydro_compact` BandSpec；随机种子 `20260831`。
- 训练/选择：仅 train/val；主选择指标为验证集像元级 `pixel_micro_mae`，不按事件选择模型。
- 旧结果未改动；本轮新产物仅位于 `runs/optimization/hydrov13_2`、`artifacts/optimization/hydrov13_2`。
- 变更前状态、HEAD、环境、pytest、旧 checkpoint 与旧验证摘要见 `prechange_status.json`。

## 诊断驱动的修改

1. 旧 Graph-KAN 诊断发现输入特征量纲/占用不均、部分描述子 knot 占用塌缩，且初始 task-only KAN 梯度为零；例如 barrier/distance 特征最大 knot 占用约 0.77–1.00，旧 graph update/input 约 5.5e-4–6.5e-4。输出见 `kan_diagnostics.json`、`kan_feature_occupancy.csv`、`kan_head_summary.csv`。
2. 旧 checkpoint 的 gate 均值约 0.44–0.51、熵 0.683–0.691，未出现 0/1 饱和；spline/base 比为 0.21–0.95，说明部分 head 已有 spline 贡献但图残差过弱。新 v13.2 首个 backward 的 edge-KAN 梯度范数为 `4.95e-5`（非零），gamma 为 0.02125–0.02142，descriptor bounded=true，graph update/input=0.00325。
3. 新 `HydroEdgeKAN` 只让 KAN 处理五个物理边特征：`signed_dz`、真实距离归一化坡度、相对障碍、局部起伏、邻居距离；八邻域对角距离使用 `sqrt(2)`。
4. 地形亲和度、观测置信度、潜变量兼容性分开建模；严格 valid edge、置信度加权均值和 confidence amplitude；输出采用 identity + residual graph update。
5. KAN 使用 train-only robust center/scale、一次固定 `[−1,1]` 映射、feature-wise spline、可学习 base/spline scale；残差强度为非负有界 `0.25*sigmoid(raw_gamma)`，初值有效强度 0.02。
6. 物理约束代码仅保留有容差的弱 terrain-order prior；本轮 baseline-loss 控制实验仍保留原 WSE 项以隔离架构变量，最终报告不把它解释为水动力方程。task-adaptive loss 消融另行运行但未用于最终。

## KAN 曲线与损失交互

- 最终 checkpoint 导出的 20 条（4 heads × 5 features）曲线均提供 base、spline、total 和一/二阶有限差分；平均绝对 spline contribution `0.02598`，最大绝对二阶差分 `2.27169`，见 `kan_curves/curve_summary.json`。
- 8 个 train batch 的共享 fusion 参数梯度诊断：uncertainty/depth 梯度比 `3.04`、bias/depth `1.39`、log/depth `0.266`、WSE/depth `0.198`、gradient/depth `0.160`；所有值有限。该结果支持将 uncertainty 与主深度梯度解耦，并拒绝强 bias/WSE 叠加。

## 受控实验（验证集）

| 候选 | 最佳 epoch | pixel MAE | RMSE | P90 AE | bias | 深水 MAE / bias (>2.14 m) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧 v13 基线复核 | 36 | 0.47345 | 0.71759 | 1.09090 | -0.19825 | 2.03669 / -2.02819 | reference |
| KAN-A，grid4，原损失 | 30 | 0.49402 | 0.76039 | 1.20471 | -0.29654 | 2.15268 / -2.15128 | 保留架构，短训不胜基线 |
| MLP diagnostic | 21 | 0.50545 | 0.76574 | 1.24182 | -0.30918 | 未记录 | 劣于 KAN-A，故不运行 KAN-B |
| task-adaptive loss | 11 | 0.59385 | 0.88658 | 1.46317 | -0.36569 | 未记录 | 损失消融失败 |
| v13.2 final raw | 39 | **0.48025** | **0.73123** | **1.13014** | **-0.21622** | **2.03061 / -2.02953** | 研究候选 |
| v13.2 final EMA | 39 | 0.49833 | 0.77438 | 1.25036 | -0.33734 | 2.20357 / -2.20332 | 不选 |

完整机器可读表：`candidate_summary.csv`；最终选择依据：`final_decision.json`。

KAN-B（grid6/其他单因素）未运行：MLP diagnostic 没有优于 KAN-A，预设触发条件不满足。final 使用 KAN-A 的 legacy loss 做 120 epoch 上限长训；task-adaptive `Loss-simple` 已实际运行但验证明显变差，因此没有把它强行替换为默认目标。

## 资源与稳定性

- 最终模型参数：5,140,461；BF16 AMP；RTX 5090。
- batch=4 吞吐约 108.65 samples/s；forward latency 0.03681 s；forward+backward 峰值显存 12,090,209,792 B。
- 梯度有限、无 AMP overflow；profile 见 `final_profile.json`。
- 全量回归：106 passed，0 failed，0 skipped，3 warnings；汇总见 `verification_summary.json`。

## 产物索引

- 模型代码：`models/hydro_edge_kan.py`、`models/pa_hydrokan_v13_2.py`、`models/kan_layers.py`。
- 损失与物理约束：`losses/task_adaptive_depth_loss.py`、`losses/composite_loss.py`、`losses/physics_losses.py`。
- 诊断工具：`tools/analyze_kan_behavior.py`、`tools/analyze_loss_interactions.py`、`tools/export_kan_curves.py`。
- KAN 曲线：`artifacts/optimization/hydrov13_2/kan_curves/curves.csv` 与 `curve_summary.json`。
- 图边 train-only 统计：`graph_edge_train_stats.json`。
- 最终 checkpoint：`runs/optimization/hydrov13_2/final/best_raw.pth`；raw/EMA 验证摘要分别位于 `final_validation_raw`、`final_validation_ema`。
- 一个 train 样本的 v13.2 CUDA 推理和 GeoTIFF/可视化导出 smoke 位于 `runs/optimization/hydrov13_2/final/infer_smoke`。

实际最终 loss（legacy-control）为
`L = 1.0 L_depth(sample-depth-bin) + 0.5 L_log + 1.0 L_aux + 0.1 L_unc(t=8,warmup=10) + 0.02 L_gradient(t=5,warmup=10) + 0.02 L_WSE(t=5,warmup=15) + 0.2 L_nnPU(t=5,warmup=10) + 1e-6 L_KAN`；最终 raw 只在 val 选择。作为失败消融，Loss-simple 实际采用 `metric Huber(β=0.50)+0.15 log1p+soft balance[0.5,3]+α_under=0.15+aux=0.05+terrain-order=0.01+detached uncertainty=0.05+KAN=5e-4`，没有虚构未运行结果。

## 限制与后续

复现命令（均不使用 test）：
`conda run --no-capture-output -n flood-depth python tools/train.py --config configs/pa_hydrokan/subset150_v13_2_final.xml --output runs/optimization/hydrov13_2/final --device cuda --seed 20260831`

当前主要瓶颈仍是深水尾部的负偏差和跨地形泛化；由于 v13.2 尚未超过旧基线，不应替换生产模型。下一轮如继续研究，应优先针对深水标注稀疏性和边缘支持误差设计小规模、预注册的尾部校准实验，并继续只用 val 做选择，最终一次性报告 test。未完成内容：KAN-B 未触发未运行；test 评价按本轮边界明确未运行；外部对比模型不在本轮范围内。
