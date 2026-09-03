# PA-HydroKAN-S1-v14 工程报告

## 结论摘要

已在 `/home/whu/桌面/myData/Flood_depth/subset1000` 上完成真正的 S1-only 路径：数据集、模型、训练、评估、推理和 profile 均由统一的 `ModelInputSpec` 驱动，S2 文件不存在时仍可运行，S1-only 样本不包含任何 `s2_*` 键。

当前证据**不能证明不使用 Sentinel-2 后更好**。已执行的候选训练是受控短预算实验（2 个 epoch、每个 epoch 4 个 train batch 和 4 个 val batch），最佳 S1-only 候选的验证集事件层级综合 MAE 为 `0.443277`，旧 v13 在相同 S1 支持区域上的验证集结果为 `0.356091`。因此目前应保留 S1-only 实现并继续做完整预算收敛实验，但不能把当前预实验结果写成性能提升。

## 1. 实施范围

- 新模型：`pa_hydrokan_s1_v14`。
- 新输入模式：`s1_terrain`。
- 数据集：`subset1000`，patch size 256，terrain metric 20 m。
- S1-only active groups：`label`、`masks`、`s1_t1`、`s1_t2`、`s1_change`、`s1_qa`、`terrain`。
- inactive groups：`s2_t1`、`s2_t2`、`s2_change`、`s2_qa`。
- 保留旧 `pa_hydrokan`、v13、v13.1、v13.2 的输入和 checkpoint 兼容路径；新模型不会改变旧模型的默认输入契约。

## 2. 无 S2 泄漏设计

S1-only 数据集只遍历 active groups，并在 `__getitem__` 中只读取这些 group。S1-only 样本和 validity 输出不会创建 S2 字段；S2 不再被读取后置零，也不参与日期、QA、可靠性或模态融合计算。

已完成的 IO 验证：

- `artifacts/optimization/hydrokan_s1_v14/io_profile_s1.json`：train/val 各读取 16 个样本、4 个 batch，`s2_opened: false`，`unexpected_s2_groups: []`。
- 读取计数只包含 `label`、`masks`、S1 三个影像 group、`s1_qa` 和 `terrain`。
- 定向测试删除可选 S2 group 后仍能加载和取样，并通过 monkeypatch 断言任何 S2 `_read` 都不会发生。
- S1-only reliability schema 只有 6 个命名通道：`s1_event_observation_count_z`、`s1_event_day_z`、`s1_available`、`dem_available`、`event_duration_log_scaled`、`s1_day_missing`；没有跨传感器日期差。

## 3. 模型与损失实现

- `sar_state_change_encoder.py`：共享 pre/event SAR 编码器、独立 change 编码器、angle FiLM、内部 `event-pre`/绝对差、外部 change、masked-softmax 权重和 QA quality gate；不使用 `pre*event`。
- `sar_terrain_fusion.py`：SAR 主流加小幅 terrain residual gate，不使用模态权重或模态熵。
- `sar_hydro_decoder.py`：decoder widths 为 `[96, 64, 48, 32]`，支持奇数尺寸，仅保留一个 1/4 分辨率 auxiliary 输出。
- `hydro_edge_kan_s1.py`：graph scale 4 默认、8 可选；边描述符为 `signed_dz`、`edge_slope`、`relative_height`、`path_barrier`、`local_relief`、`distance`，包含静态地形亲和、观测置信度、latent compatibility 和 valid-edge gate。
- 输出支持分支默认关闭；不确定性分支与 depth backbone 解耦，并在 loss 中 detached 到 depth。
- 默认损失为 depth Huber + `0.15` log-depth + `0.05` auxiliary + `0.01` uncertainty + `1e-6` KAN；WSE 默认关闭，后续顺序项为 `0.005`，underprediction/exceedance/PU/gradient/laplacian/bias 项默认关闭。

## 4. 验证与 profile

完整回归测试最终结果：`121 passed, 2 skipped, 3 warnings`。

S1-only 定向验证覆盖：无 S2 文件读取、无 S2 样本键、缺失可选 S2 group、模型 forward/backward、奇数尺寸 decoder、registry、有限梯度和旧 checkpoint 兼容。

最终 profile（CPU、batch size 4）：3,006,322 个参数；forward/backward 梯度有限；depth、uncertainty 和 KAN diagnostics 均有限。KAN diagnostics 中 valid-edge fraction 为 `0.946486`，static topographic affinity mean 为 `0.935093`，observation confidence mean 为 `0.286990`。

## 5. 受控实验结果

指标为事件层级综合 MAE，越低越好；S1 候选均为短预算结果。

| 实验 | split / support | event hierarchical composite MAE | pixel micro MAE | 说明 |
|---|---|---:|---:|---|
| old v13 | val / common S1 support | 0.356091 | 0.395917 | 旧基线重评估 |
| S1 minimal | val / native S1 support | 0.464487 | 0.508631 | graph off、aux off |
| S1 v14 KAN scale4 | val / native S1 support | **0.443277** | 0.504716 | 当前短预算最佳 S1 候选 |
| S1 v14 final | val / common S1 support | 0.472168 | 0.510775 | final 配置短预算 checkpoint |
| S1 v14 final | test / native S1 support | 0.372642 | 0.405415 | 仅作测试记录，不与 val 基线横向比较 |

old v13 的 native full support 与 common S1 support 在本次重评估中像素数相同（732,718），但候选仍采用 common support 作为公平对照。当前最佳 S1 候选相对该基线高 `0.087187` MAE；这只说明短预算尚未显示优势，不说明 S1-only 在完整训练下必然更差。

## 6. 可复现实验入口与产物

- 主配置：[subset1000_s1_v14_final.xml](../configs/pa_hydrokan/subset1000_s1_v14_final.xml)
- 最小配置：[subset1000_s1_minimal.xml](../configs/pa_hydrokan/subset1000_s1_minimal.xml)
- graph scale 4 配置：[subset1000_s1_v14.xml](../configs/pa_hydrokan/subset1000_s1_v14.xml)
- graph scale 8 可选配置：[subset1000_s1_v14_scale8_optional.xml](../configs/pa_hydrokan/subset1000_s1_v14_scale8_optional.xml)
- 运行目录：`runs/optimization/hydrokan_s1_v14/`
- 结果目录：`artifacts/optimization/hydrokan_s1_v14/`
- 候选汇总：[candidate_summary.json](../artifacts/optimization/hydrokan_s1_v14/candidate_summary.json)
- IO profile：[io_profile_s1.json](../artifacts/optimization/hydrokan_s1_v14/io_profile_s1.json)
- 模型 profile：[final_profile.json](../artifacts/optimization/hydrokan_s1_v14/final_profile.json)
- KAN 诊断：[kan_diagnostics.json](../artifacts/optimization/hydrokan_s1_v14/kan_diagnostics.json)
- 最终验证摘要：[verification_summary.json](../artifacts/optimization/hydrokan_s1_v14/verification_summary.json)

## 7. 后续实验边界

正式 120 epoch 全数据训练、scale 8 训练和多 seed 统计尚未执行；当前结论是“工程路径已通过验证，性能结论仍为 provisional”。下一步应在同一 split、同一 common support、同一评估脚本下完成 full-budget S1-only 与旧 v13 的 paired re-evaluation，再决定是否替换默认模型。
