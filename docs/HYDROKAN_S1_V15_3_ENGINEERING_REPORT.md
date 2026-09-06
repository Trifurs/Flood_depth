# PA-HydroKAN-S1 V15.3：严格筛选与最终工程报告

## 结论先行

本轮没有接受新的模型版本。严格复训的无光学 V15 baseline raw validation MAE 为 `0.38763855`；所有 V15.3 候选均未在相同的 45 epoch、batch 12、seed `20260904`、S1-only 验证协议下超过该值。

最接近的候选是 **Accuracy-first V15 + Physics-A**：MAE `0.38833173`，比 fresh matched baseline 高 `0.00069318`（`+0.1788%`）。按照“新版本必须优于 V15 无光学版本”的硬门槛，该候选被拒绝；最终决策为：

```text
retain_current_best_v15
```

本轮没有使用 test split，也没有读取或使用 Sentinel-2 输入。机器可读结论见 `artifacts/optimization/hydrokan_s1_v15_3/final_decision.json`。

## 1. 范围与输入审计

- 数据根目录：`/home/whu/桌面/myData/Flood_depth`。
- 当前实验数据：`subset1000_s1_only`，`input_mode=s1_terrain`。
- 连续输入：`s1_t1`、`s1_t2`、`s1_change`、`terrain`。
- Sentinel-2 的 `s2_t1`、`s2_t2`、`s2_change`、`s2_qa` 全部为 inactive；训练深度扫描记录 `s2_files_opened=0`。
- 所有筛选、bootstrap、反事实和梯度诊断都只使用 validation；`test_split_used=false`。

V15.3 的目标是针对云雾影响下的 S1-only 输入提升稳定性，保留 Graph、KAN 和 weak physics 的可审计路径，同时先解决 V15.2 暴露出的 seed instability 和图消息过强风险。没有修改原始栅格数据或数据划分。

## 2. 工程修改

| 模块 | 主要修改 |
| --- | --- |
| Graph message | `models/hydro_edge_kan_v15_1.py` 增加 `linear/tanh/rational` 有界消息、scale 校验、extreme-preservation gate 和诊断量；V15.2 wrapper 透传这些参数。 |
| V15.3 模型 | 新增 `models/pa_hydrokan_s1_v15_3.py`：stride-4 Graph+KAN、rational message、neutral gate initialization、event-main S1 path、极值保留。 |
| Physics-C | `losses/physics_losses.py` 增加 `wse_consistency` 模式；`losses/composite_loss.py` 透传 tolerance、softplus temperature 和 maximum barrier。 |
| 注册与测试 | `utils/registry.py` 注册 V15.3；新增 V15.3 stability 和 Physics-C 单元测试；counterfactual 工具支持 V15.3。 |
| 实验汇总 | 新增 `tools/summarize_v15_3_screening.py`，统一输出候选、Physics、profiling、bootstrap、seed 状态和最终决策。 |

所有新增候选仍保留 Graph + KAN + weak physics；没有通过删除模块来制造不公平结果。Accuracy-first 分支使用原 V15 结构加物理项，作为对“新结构本身是否导致退化”的控制。

## 3. CUDA、显存与共同协议

CUDA 已确认可用：NVIDIA GeForce RTX 5090，PyTorch `2.11.0+cu130`，BF16 supported。真实栅格 forward/backward profile 中 batch 16 对 baseline 和 V15.3 stability route 均 OOM；因此选用两条路径共同可稳定运行的最大 batch 12。

| 项目 | 统一值 |
| --- | ---: |
| maximum epochs | 45 |
| minimum epochs | 30 |
| screening seed | 20260904 |
| batch / accumulation | 12 / 1 |
| AMP | BF16 |
| optimizer | AdamW, LR `2.5e-4`, WD `1e-4` |
| scheduler | cosine with 5-epoch warmup |
| sampler | event-balanced, no replacement |
| augmentation | horizontal/vertical flip `0.5`，其余关闭 |
| primary selection | raw canonical validation pixel-micro MAE |
| test split | 未使用 |

Profile 摘要：matched V15 参数量 `3,502,834`、batch-12 peak `29,066,153,472 B`、forward `130.605 samples/s`；V15.3 stability 参数量 `2,574,998`、peak `25,254,756,864 B`、forward `156.003 samples/s`。效率改善没有转化为精度改善，因此不能成为接收理由。

## 4. Fresh matched baseline

历史三种子 Matched V15 参考均值为 MAE `0.38163103 ± 0.00754728`，来自 V15.2 严格记录。为避免跨 epoch/早停协议比较，本轮又用同一 V15 结构、同一 45-epoch 预算和同一 seed `20260904` 独立复训 baseline，作为所有 V15.3 筛选的首要门槛。

| 权重 | best epoch | MAE | RMSE | P90 | bias | deep MAE (>=0.5 m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh matched raw | 37 | 0.387639 | 0.708808 | 0.957855 | -0.191469 | 0.699136 |
| fresh matched EMA | 44 | 0.395181 | 0.728324 | 0.998999 | -0.222150 | 0.732256 |

EMA 仅作为独立诊断，不替代预先指定的 raw 选择权重。历史记录中的 `0.40226` 属于不同训练预算/epoch 的非匹配结果，不能用来降低本轮验收门槛。

## 5. 候选对比

以下均为同一 validation protocol 的 independent evaluation。`ΔMAE` 定义为 candidate minus fresh matched baseline，正值表示变差。

| 候选 | raw epoch | raw MAE | raw RMSE | raw P90 | raw bias | EMA MAE | deep raw MAE | ΔMAE | 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fresh matched V15 baseline | 37 | 0.387639 | 0.708808 | 0.957855 | -0.191469 | 0.395181 | 0.699136 | 0 | reference |
| V15.3 stability Graph+KAN | 36 | 0.393238 | 0.716671 | 0.995337 | -0.236987 | 0.402225 | 0.745073 | +0.005599 | reject |
| V15.3 stability + Physics-A | 36 | 0.393009 | 0.716348 | 0.995223 | -0.236203 | 0.402072 | 0.744050 | +0.005370 | reject |
| V15.3 stability + Physics-C | 36 | 0.393300 | 0.717108 | 0.994386 | -0.236845 | 0.402200 | 0.744919 | +0.005662 | reject |
| Accuracy-first V15 + Physics-A | 34 | **0.388332** | **0.699345** | **0.933840** | **-0.158892** | 0.403152 | **0.675901** | **+0.000693** | **reject MAE gate** |
| Accuracy-first V15 + Physics-C | 34 | 0.388791 | 0.700460 | 0.938488 | -0.159605 | 0.403301 | 0.676759 | +0.001152 | reject |
| Accuracy-first V15 + Physics-A, lambda=0.001 | 34 | 0.389302 | 0.699383 | 0.933314 | -0.158444 | 0.403171 | 0.677124 | +0.001664 | reject |

Accuracy-first Physics-A 在 RMSE、P90、overall bias 和 deep MAE 上出现局部改善，但主指标 MAE 仍劣于 fresh baseline；这类次级指标改善不足以绕过预先声明的精度替代门槛。其 paired validation 的 sample-unit MAE win rate 仅 `38.20%`，event-unit 仅 `34.62%`。

## 6. Physics-A 与 Physics-C

两种物理项均真实参与训练，effective weight 非零，但都没有通过主 MAE 门槛。

| 候选 | mode | lambda | active pair fraction | mean violation (m) | effective weight | raw MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stability Physics-A | terrain_order_margin | 0.0025 | 0.2696 | 0.00727 | 0.0025 | 0.393009 |
| stability Physics-C | wse_consistency | 0.0025 | 0.9946 | 0.02906 | 0.0025 | 0.393300 |
| accuracy Physics-A | terrain_order_margin | 0.0025 | 0.2696 | 0.00825 | 0.0025 | 0.388332 |
| accuracy Physics-C | wse_consistency | 0.0025 | 0.9946 | 0.02986 | 0.0025 | 0.388791 |
| accuracy Physics-A | terrain_order_margin | 0.0010 | 0.2696 | 0.00836 | 0.0010 | 0.389302 |

Physics gradient probe（真实 batch、BF16、无 optimizer step）显示：Physics-A loss `0.0233356`，active fraction `0.4473`，physics gradient norm `1.0848e-05`，监督深度梯度 norm `0.0084598`，cosine `0.05975`，所有梯度 finite。KAN probe 中 spline gradient norm `1.8393e-04`、base gradient norm `2.9751e-05`、graph gamma gradient norm `1.4832e-04`，均 finite/nonzero。上述结果证明机制没有“只写进 config、实际不求梯度”的问题，但不证明机制提升了泛化精度。

## 7. Graph / KAN 反事实诊断

在 V15.3 stability Physics-A 的 raw checkpoint 上进行，full 与独立 evaluation 一致性在 `5e-6` 内。正 Δ 表示相对 full 变差；该诊断只用于机制审计，不用于调 test 或选择最终模型。

| intervention | MAE | ΔMAE vs full | deep MAE | 解释 |
| --- | ---: | ---: | ---: | --- |
| full | 0.393009 | 0 | 0.744050 | 当前结构 |
| graph off | 0.391647 | -0.001362 | 0.738813 | 当前稳定结构中的 graph residual 反而略伤 MAE |
| spline off | 0.392668 | -0.000341 | 0.743077 | spline 路径非零，但本 checkpoint 未带来净提升 |
| base off | 0.393150 | +0.000141 | 0.744392 | base 分支小幅有益 |
| constant gate | 0.394054 | +0.001045 | 0.746167 | 学习 gate 有效 |
| shuffled terrain | 0.392892 | -0.000117 | 0.743686 | terrain 因果贡献仍偏弱，不能宣称为稳定增益 |

这组结果与 V15.2 早期 Graph 诊断不同：V15.3 的 neutral gate 和 bounded message 解决了数值风险，但当前 graph message 仍未形成精度收益。因此 V15.3 不能以“结构更稳定”推断“精度更高”。

## 8. Paired bootstrap

Accuracy-first Physics-A 与 fresh baseline 在同一 validation 样本上配对，10,000 次 bootstrap；没有使用 test split。

| 重采样单位 | ΔMAE（candidate-baseline） | 95% CI | candidate win rate | ΔRMSE | 95% CI |
| --- | ---: | --- | ---: | ---: | --- |
| sample（89 units） | +0.000693 | [-0.005835, 0.009154] | 38.20% | -0.009462 | [-0.015785, -0.000938] |
| event（26 units） | +0.000693 | [-0.007979, 0.019195] | 34.62% | -0.009462 | [-0.017359, 0.010442] |

MAE 点估计仍是候选变差；RMSE 的改善不改变主 MAE gate。区间跨零时只能说明单 seed 的不确定性较大，不能据此宣称候选稳定优于 baseline。

## 9. 验证结果与多 seed 决策

完整回归测试：

```text
conda run --no-capture-output -n flood-depth python -m pytest -q
169 passed, 2 skipped, 2 warnings
```

CUDA smoke、BF16 finite-gradient profile、S1 input contract、Physics-C 单元测试、Graph/KAN counterfactual 一致性和独立 raw/EMA evaluation 均已完成。旧 modality-dropout 配置产生的两条 warning 是兼容映射提示，不是失败。

根据预先的 gate，只有筛选阶段 raw MAE 严格优于 fresh matched baseline 的候选才进入三种子 final stage。本轮 passing candidates 为空，因此没有对失败候选额外运行三种子训练，也没有人为制造一个“final candidate”统计。历史 Matched V15 的三种子参考仍保留：raw MAE `0.38163103 ± 0.00754728`（seeds `20260904/05/06`）。

这同时满足用户提出的控制变量要求：对比使用同一 epoch 上限、最低 epoch、batch、AMP、数据划分和 selection rule；`0.40226` 等不匹配历史数字不会被用来接受新模型。

## 10. 结论与后续边界

1. 接受结果：无；保留当前 V15 无光学模型。
2. V15.3 代码和诊断结果保留为研究分支，不作为部署 checkpoint。
3. 不接受 Accuracy-first Physics-A，尽管其 RMSE/P90/deep MAE 改善，因为整体 raw MAE 仍高 `0.1788%`。
4. 不接受 stability Graph+KAN 或 Physics-C；它们的 raw MAE 分别比 baseline 高约 `1.444%`、`1.385%` 和 `1.461%`（Physics-A/C 以对应候选计）。
5. 后续若继续研究，应优先处理 graph residual 的因果贡献、SAR change/event reliability 和深水欠估计；不得用本轮 test split（本轮未使用）或单 seed 次级指标放宽主门槛。

## 11. 关键工件

| 内容 | 路径 |
| --- | --- |
| 候选逐项汇总 | `artifacts/optimization/hydrokan_s1_v15_3/candidate_summary.json` / `.csv` |
| 最终拒绝/保留决策 | `artifacts/optimization/hydrokan_s1_v15_3/final_decision.json` |
| Physics-A/C 对比 | `artifacts/optimization/hydrokan_s1_v15_3/physics_comparison.json` |
| 多 seed 状态 | `artifacts/optimization/hydrokan_s1_v15_3/multiseed_summary.json` / `.csv` |
| CUDA profile | `artifacts/optimization/hydrokan_s1_v15_3/final_profile.json` |
| paired bootstrap | `artifacts/optimization/hydrokan_s1_v15_3/paired_bootstrap.json` |
| Graph/KAN 反事实 | `artifacts/optimization/hydrokan_s1_v15_3/graph_counterfactuals.json`、`kan_counterfactuals.json` |
| seed instability | `artifacts/optimization/hydrokan_s1_v15_3/seed_instability_analysis.json` |
| 总验证 | `artifacts/optimization/hydrokan_s1_v15_3/verification_summary.json` |
| fresh matched baseline raw checkpoint | `runs/optimization/hydrokan_s1_v15_3/matched_current_best/best_raw.pth` |
| 本报告 | `docs/HYDROKAN_S1_V15_3_ENGINEERING_REPORT.md` |
