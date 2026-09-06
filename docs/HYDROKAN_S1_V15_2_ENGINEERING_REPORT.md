# PA-HydroKAN-S1 V15.2：工程审计与严格验证报告

## 结论先行

本轮没有产生可接受的 V15.2 精度替代模型。最终受约束候选（S1 + terrain + Graph + KAN + weak physics）在严格匹配的三种子验证中，raw checkpoint 的平均 pixel-micro MAE 为 `0.387698 ± 0.019166`，严格 Matched V15 为 `0.381631 ± 0.007547`。候选高出 `0.006067 m`，即 **MAE 恶化 1.590%**；P90、深水 MAE 和 bias 也都更差。因此其状态是：

`reject_candidate_as_accuracy_replacement`

这个结论只使用 validation；没有读取、调参或报告 test split。所有最终运行均为 `s1_terrain`，S2 T1/T2/change/QA 组均处于 inactive 状态。

## 1. 修改文件

本轮保留原有 Graph、KAN 和弱物理约束的前提下，主要新增或修改了以下工程文件。

| 范畴 | 文件 | 作用 |
| --- | --- | --- |
| S1 主干与模型 | `models/s1_hydrology_backbone_v15_1.py`、`models/pa_hydrokan_s1_v15_1.py`、`models/pa_hydrokan_s1_v15_2.py` | S1-only SAR/terrain 主路径、时相变化修复、Graph-A/Graph-B 接入与严格输入检查。 |
| Graph/KAN | `models/hydro_edge_kan_v15_1.py`、`models/hydro_edge_kan_v15_2.py`、`models/kan_layers.py` | 边特征、反事实开关、base/spline 分解、有效边掩膜、feature-wise KAN 诊断与有效 spline 正则。 |
| 损失与训练 | `losses/frozen_soft_depth_balance.py`、`losses/soft_depth_balance.py`、`losses/task_adaptive_depth_loss.py`、`losses/composite_loss.py`、`losses/physics_losses.py`、`tools/train.py`、`utils/ema.py` | 冻结深度权重、log Huber beta、EMA fallback、弱物理项及真实梯度诊断。 |
| 运行与验证 | `tools/summarize_v15_2_candidates.py`、`tools/summarize_v15_2_final.py`、`tools/build_v15_2_verification_summary.py`、`tools/evaluate_v15_2_kan_counterfactuals.py`、`tools/analyze_v15_1_kan.py`、`tools/probe_v15_2_kan_gradient.py` | 45-epoch 筛选汇总、三种子统计、paired bootstrap、Graph/KAN 反事实、KAN 梯度审计。 |
| 配置 | `configs/pa_hydrokan/subset1000_s1_v15_2_*.xml` | Matched V15、simple/Graph-A、KAN-map、Graph-B、Physics-A/B、组合、损失/增强诊断和最终三种子配置。 |
| 测试 | `tests/test_v15_2_correctness.py`、`tests/test_v15_2_weak_physics.py`、`tests/test_hydro_edge_kan_v15_2.py`、`tests/test_v15_2_kan_counterfactuals.py` 及相关 V15.1 测试 | S1-only、冻结权重、时相变化、log beta、EMA、Graph/KAN、物理项和 checkpoint 约束。 |

最终回归命令为：

```bash
conda run --no-capture-output -n flood-depth python -m pytest -q --ignore=tests/test_dataset_loading.py
```

结果：`165 passed, 2 skipped, 2 warnings`。两条 warning 都是旧 `modality_dropout_probability` 兼容映射提示，不是失败。

## 2. 严格 baseline

历史无光学 V15 的 `0.40226` 来自不同训练预算/早停位置，不能作为本轮“是否升级”的严格 baseline。它只保留为历史上下文；本轮结论使用重新训练的 Matched V15。

筛选阶段的 Matched V15 与所有候选固定为 seed `20260904`、BF16、真实 batch `12`、accumulation `1`、AdamW（LR `2.5e-4`，WD `1e-4`，KAN LR multiplier `0.5`）、cosine + 5 epoch warmup、event-balanced sampler、水平/垂直翻转各 `0.5`、45 epochs（minimum 30），并按 raw canonical validation MAE 选 checkpoint。

| 项目 | Matched V15（45 epoch 严格筛选） |
| --- | --- |
| seed / AMP / batch | `20260904` / BF16 / `12` |
| best raw epoch（zero-based） | `37` |
| MAE / RMSE / P90 / bias | `0.387639 / 0.708808 / 0.957855 / -0.191469` |
| deep MAE / deep bias（>= 0.5 m） | `0.699136 / -0.635606` |
| 参数量 | `3,502,834` |
| forward samples/s | `129.062` |
| forward+backward peak memory | `29,065,317,888 B` |

最终三种子阶段仅把共同预算改为 maximum `100`、minimum `35`、patience `20`；候选与 Matched V15 的 `training`、`optimizer`、`scheduler`、`dataset`、`supervision` 段完全一致。候选唯一的机制差异是 `lambda_phys=0.0025`；Matched V15 为精确 `0`。机器可读验证见 `verification_summary.json`。

## 3. Correctness fixes

- **冻结 soft-depth balance**：从 canonical train positive depth 的 `5,408,182` 像元一次性建立权重曲线，工件 SHA-256 为 `9d27f76c…097ff4e`。运行时只做 `target -> frozen weight` 插值，绝不根据 minibatch 再归一化；train 均值为 `1.0`，实际范围 `0.6764–3.0`，配置边界为 `[0.5, 3.0]`。
- **时相 dropout**：共享 pre/event 状态编码路径内部关闭独立随机 dropout，随机性移到状态/变化融合后；因此相同 pre/event 和零变化在 train mode 下不会制造伪 internal change。
- **log-depth Huber beta**：`log_depth_huber_beta` 已真实传入 task-adaptive log loss；无效 beta 会报错。
- **EMA fallback**：旧 checkpoint 缺少 EMA 状态时，EMA 从已载入 raw model 初始化；raw 与 EMA 均可独立评估。最终协议预先指定 raw 为主报告权重，三种子 EMA 工件仍全部保存。
- **QA/reliability 去重**：使用单一 S1 reliability conditioner，`s1_qa_channels=0`，避免安全 QA 信息以多个并行路径重复注入。
- **depth head 初始化**：由 canonical train positive-depth median `0.24 m` 校准 softplus head bias，而非使用任意常数；初始化统计和冻结深度权重均是 train-only 工件。

S1-only 检查显示连续输入仅为 `s1_t1`、`s1_t2`、`s1_change`、`terrain`；`s2_t1`、`s2_t2`、`s2_change`、`s2_qa` 均 inactive。训练深度扫描记录 `s2_files_opened=0`。

## 4. 模型结构、KAN 与其诊断

### 数据流和张量形状

输入 patch 为 `B × C × 256 × 256`：S1 pre `C=2`、S1 event `C=2`、S1 change `C=3`、S1 angle conditioning `C=2`，terrain `C=2`（DSM elevation、slope）。SAR 与 terrain 金字塔的通道宽度为 `[32, 64, 128, 192]`。在 stride-4 的 `64 × 64` 特征层，`128` 通道融合特征先投影为 `64` 通道，再经过两个 32-channel head 的 HydroEdgeKAN；Graph update 回投影后以残差方式加入原特征。最后使用轻量 context/FPN decoder 恢复到 `B × 1 × 256 × 256` 的 conditional positive depth；不把 DSM 称为 DTM，也不把该模型称为 PINN 或水动力求解器。

最终候选保留原始 KAN：grid `4`、cubic spline order `3`、统一稳健映射温度 `1.5`、base scale init `0.35`、spline scale init `1.0`、有效 spline 正则 `lambda_kan=1e-6`。边关系输入为：

1. `signed_grade`；
2. `absolute_grade`；
3. `barrier_magnitude`；
4. `local_surface_complexity`。

KAN-map（feature-wise temperatures `[4.0, 1.5, 1.5, 1.5]`）的 45-epoch MAE 为 `0.385318`，差于 simple-fixed 的 `0.375174`，因此按预先规则没有继续运行 grid `4 -> 6`；这不是遗漏实验。

最终候选 seed `20260904` 的验证诊断如下。四个 feature 均有 `2,848,356` 个有效边值。

| feature | 四个 knot interval occupancy | boundary saturation |
| --- | --- | ---: |
| signed grade | 598280 / 703432 / 948364 / 598280 | 24.539% |
| absolute grade | 0 / 1455090 / 590336 / 802930 | 13.321% |
| barrier magnitude | 0 / 1671102 / 572258 / 604996 | 7.165% |
| local surface complexity | 0 / 1485946 / 616320 / 746090 | 9.131% |

整体边界饱和率为 `13.311%`，其中 signed grade 的 `24.539%` 明显偏高，是后续若继续研究 KAN 时应优先处理的问题。base RMS 为 `0.641275`、spline RMS 为 `0.356612`，spline/base RMS ratio 为 `0.586361`，说明 spline 不是数值零分支。

为避免仅以单元测试断言梯度存在，额外对最终 checkpoint 做了一个 CUDA/BF16、单训练 batch、无 optimizer step 的审计（objective epoch 24，physics 权重已达到 `0.0025`）：spline coefficient gradient norm 为 `0.0085353`，base gradient norm 为 `0.0013984`，两者均 finite；graph raw-gamma gradient norm 为 `0.0102775`。这只证明 KAN 接收实际训练梯度，不等价于证明它带来多种子平均精度提升。

### KAN/Graph 反事实

下表为最终候选 seed `20260904` 的 raw validation 反事实，正 ΔMAE 表示比 full 更差；它们与 full checkpoint 指标一致到 `5e-6` 容差内。

| intervention | MAE | ΔMAE | ΔP90 | Δbias |
| --- | ---: | ---: | ---: | ---: |
| full | 0.370606 | 0 | 0 | 0 |
| graph-off | 0.374518 | +0.003913 | -0.001659 | +0.023911 |
| spline-off | 0.370756 | +0.000151 | -0.002918 | +0.004511 |
| base-off | 0.371065 | +0.000460 | +0.001608 | -0.006013 |
| constant-gate | 0.376214 | +0.005609 | +0.024186 | -0.031546 |
| shuffled-terrain | 0.371163 | +0.000557 | -0.000645 | +0.003557 |

因此，Graph 和 KAN spline 都不是装饰性零通路：Graph-off 恶化 `0.003913` MAE，spline-off 恶化 `0.000151` MAE，base-off 恶化 `0.000460` MAE。KAN 的增益较小，且这是单 checkpoint 诊断；它不能抵消最终三种子主比较的拒绝结论。

## 5. Graph

最终候选采用 **Graph-A skip-only**，位于 stride-4 特征层：节点间距 `80 m`，正交邻居距离 `80 m`，对角距离 `113.137 m`，图特征大小 `64 × 64`。Graph-B 额外把 pooled graph residual 写入 bottleneck；其固定 45-epoch MAE 为 `0.406796`，明显差于 Graph-A/simple-fixed 的 `0.375174`，故未进入最终组合。

最终候选 seed `20260904` 的 Graph 审计值为：有效边比例 `0.963886`；最终 gate 分布 `mean/p05/p50/p95 = 0.430957 / 0.162691 / 0.487296 / 0.612241`；gamma 均值 `0.033541`；graph input RMS `0.709278`；update RMS `0.064703`；update/input RMS `0.091273`。这与 graph-off 和 constant-gate 反事实共同表明局部空间传播确实被使用。

## 6. Physics loss

最终受约束候选使用 Physics-A `terrain_order_margin`，不是 PINN 或流体求解器。对一个满足资格的局部邻接对 `(i, j)`，令 `Δz = z_j-z_i`，预测水深为 `d`，其方向性超额为：

```text
s = sign(Δz) · (d_j - d_i) - τ_d
penalty = T · softplus(s / T)
```

其中 `τ_d=0.02 m`、`T=0.05 m`。只有两端都是 canonical labelled-positive、DEM 和 S1 有效、位于图像边界内、平均 local relief 不超过 `12 m`、且 `0.05 <= |Δz| <= 0.75 m` 的 pair 会参与；其权重还乘以 `exp(-mean_relief / 12)`。这是一条地形一致性的弱先验，不假定 DSM 是 DTM，不宣称质量守恒或流量路由。

`lambda_phys=0.0025`，从 epoch `15` 开始，10 epoch 线性 warmup（epoch 15 为 `0.00025`，epoch 24 起为 `0.0025`）。三种子所有 physics-active epochs 的平均实际统计是：

| 指标 | 值 |
| --- | ---: |
| effective physics weight | 0.002207 |
| active pair fraction | 0.275760 |
| active pair count | 81,397.65 |
| mean / P90 violation | 0.008124 m / 0.015101 m |
| physics depth-gradient norm | 6.291e-06 |
| supervised depth-gradient norm | 2.589e-03 |
| gradient norm ratio | 约 0.243% |
| physics-vs-depth gradient cosine | 0.034522 |

物理梯度在三种子中均非零，故它不是仅在 config 中出现的死项。筛选阶段 Physics-A 的 MAE `0.374633` 略优于无物理的 simple-fixed `0.375174`，也略优于 Physics-B `0.375351`，所以选择 Physics-A。这个单 seed 迹象没有在最终严格三种子比较中转化为对 Matched V15 的净精度提升；因此只能说弱物理项**实际生效且筛选期可接受**，不能宣称其已被证明改善总体平均精度。

## 7. 所有筛选候选

以下均为 seed `20260904`、严格共同 45-epoch 筛选、raw canonical validation checkpoint。参数量和吞吐仅在已 profile 的模型中填入；`—` 表示未重复 profile，不代表运行失败。Combined 是 Physics-A 的确定性配置别名，因此不重复训练相同变量组合。

| 候选 | best epoch | MAE | RMSE | P90 | bias | deep MAE | deep bias | 参数量 | fwd samples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched V15 baseline | 37 | 0.387639 | 0.708808 | 0.957855 | -0.191469 | 0.699136 | -0.635606 | 3,502,834 | 129.062 |
| V15.1-simple-fixed / Graph-A | 32 | 0.375174 | 0.695801 | 0.921942 | -0.200893 | 0.686433 | -0.642559 | 2,575,006 | 159.679 |
| V15.2 KAN-map | 37 | 0.385318 | 0.697287 | 0.975166 | -0.201957 | 0.702694 | -0.647042 | 2,574,998 | 158.350 |
| Graph-B skip+bottleneck | 38 | 0.406796 | 0.736587 | 1.025946 | -0.227968 | 0.762503 | -0.696069 | 2,599,583 | 159.412 |
| Physics-A terrain-order | 32 | 0.374633 | 0.695576 | 0.921666 | -0.201383 | 0.686280 | -0.641625 | 2,575,006 | — |
| Physics-B barrier-aware | 32 | 0.375351 | 0.696281 | 0.924191 | -0.202672 | 0.688302 | -0.643884 | 2,575,006 | — |
| Combined Graph+KAN+Physics-A | 32 | 0.374633 | 0.695576 | 0.921666 | -0.201383 | 0.686280 | -0.641625 | 2,575,006 | — |
| Raw SAR shortcut on | 37 | 0.383925 | 0.704974 | 0.979331 | -0.225903 | 0.721422 | -0.673034 | 2,588,386 | 155.871 |
| Geometry flips off | 34 | 0.401331 | 0.749402 | 0.993570 | -0.219582 | 0.742233 | -0.672106 | 2,575,006 | — |
| Depth Huber beta = 0.25 | 37 | 0.378786 | 0.716745 | 0.951966 | -0.229693 | 0.713038 | -0.670649 | 2,575,006 | — |
| Log weight = 0.10 | 35 | 0.381049 | 0.707367 | 0.952051 | -0.218660 | 0.708265 | -0.657549 | 2,575,006 | — |
| Tail underprediction alpha = 0.10 | 37 | 0.379855 | 0.686754 | 0.937503 | -0.201017 | 0.696990 | -0.622125 | 2,575,006 | — |
| Tail underprediction alpha = 0.15 | 37 | 0.379107 | 0.715841 | 0.916607 | -0.202248 | 0.694720 | -0.637183 | 2,575,006 | — |

## 8. 多 seed 严格验证与效率

最终候选和 Matched V15 都使用种子 `20260904 / 20260905 / 20260906`、最大 100 epoch、minimum 35、patience 20。下表的 best epoch 是 zero-based checkpoint epoch，完成 epoch 数包括 epoch 0。

| variant | seed | best raw epoch | 完成 epoch 数 | MAE | RMSE | P90 | bias | deep MAE | deep bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate | 20260904 | 38 | 59 | 0.370606 | 0.684289 | 0.908748 | -0.200174 | 0.678400 | -0.631242 |
| Candidate | 20260905 | 22 | 43 | 0.408419 | 0.760225 | 1.025082 | -0.262111 | 0.783885 | -0.747285 |
| Candidate | 20260906 | 37 | 58 | 0.384069 | 0.716581 | 0.972396 | -0.245526 | 0.734734 | -0.694785 |
| Matched V15 | 20260904 | 64 | 85 | 0.380107 | 0.691995 | 0.912963 | -0.180595 | 0.676985 | -0.609918 |
| Matched V15 | 20260905 | 43 | 64 | 0.374962 | 0.710498 | 0.919421 | -0.231201 | 0.703283 | -0.667303 |
| Matched V15 | 20260906 | 35 | 56 | 0.389824 | 0.725505 | 0.980482 | -0.217443 | 0.719137 | -0.672469 |

下表格式为 `mean ± sample std [median; min, max]`。

| metric | Candidate | Matched V15 |
| --- | --- | --- |
| MAE | 0.387698 ± 0.019166 [0.384069; 0.370606, 0.408419] | 0.381631 ± 0.007547 [0.380107; 0.374962, 0.389824] |
| RMSE | 0.720365 ± 0.038109 [0.716581; 0.684289, 0.760225] | 0.709333 ± 0.016785 [0.710498; 0.691995, 0.725505] |
| P90 | 0.968742 ± 0.058253 [0.972396; 0.908748, 1.025082] | 0.937622 ± 0.037258 [0.919421; 0.912963, 0.980482] |
| bias | -0.235937 ± 0.032062 [-0.245526; -0.262111, -0.200174] | -0.209746 ± 0.026166 [-0.217443; -0.231201, -0.180595] |
| deep MAE | 0.732340 ± 0.052784 [0.734734; 0.678400, 0.783885] | 0.699801 ± 0.021291 [0.703283; 0.676985, 0.719137] |
| deep bias | -0.691104 ± 0.058109 [-0.694785; -0.747285, -0.631242] | -0.649896 ± 0.034719 [-0.667303; -0.672469, -0.609918] |

最终 CUDA/BF16 profile 证明两条路径均无 overflow、梯度均 finite。候选的效率更好，但效率不能覆盖精度验收失败。

| 指标 | Candidate | Matched V15 | 候选变化 |
| --- | ---: | ---: | ---: |
| 参数量 | 2,575,006 | 3,502,834 | -26.488% |
| forward+backward peak memory | 24,989,484,544 B (23.273 GiB) | 29,120,869,376 B (27.121 GiB) | -14.187% |
| forward samples/s | 157.285 | 130.791 | +20.257% |
| forward+backward samples/s（由 batch/time 推得） | 31.866 | 29.251 | +8.938% |

## 9. paired validation 与严格结论

paired bootstrap 使用 10,000 draws，分别以 sample（89）和 event（26）作为重采样单位。所有区间按独立训练的 seed pair 分开计算，**没有**把三组 seed 当成额外 validation samples 进行不当 pooled CI。下表的 ΔMAE 为 `Candidate - Matched V15`。

| seed | sample ΔMAE [95% CI] | sample candidate win rate | event ΔMAE [95% CI] | event candidate win rate |
| --- | --- | ---: | --- | ---: |
| 20260904 | -0.009501 [-0.022246, 0.002044] | 53.93% | -0.009501 [-0.031599, 0.007683] | 46.15% |
| 20260905 | +0.033457 [0.010385, 0.056967] | 35.96% | +0.033457 [-0.015242, 0.056686] | 46.15% |
| 20260906 | -0.005755 [-0.022088, 0.007437] | 50.56% | -0.005755 [-0.036699, 0.006837] | 65.38% |

sample 的平均 candidate win rate 为 `46.82%`，event 的平均为 `52.56%`。其中 seed `20260905` 的 sample-level CI 完全大于零，支持候选在该训练对上更差；另外两组 sample/event CI 都跨零。它们均不支持“候选稳定改善”的主张。

严格结论如下：

- **平均 MAE 是否提高？否。** 候选平均 MAE 高 `0.006067 m`，相对恶化 `1.590%`；未达到“优于 baseline”，更未达到优先目标 `>=0.5%` 相对改善。
- **95% paired bootstrap 是否支持改善？否。** 没有一个统一 pooled CI 被不当构造；逐 seed 结果中一个样本级 CI 明确支持变差，其他 CI 大多跨零。
- **P90 是否改善？否。** 候选 `0.968742`，比 baseline 高 `0.031120`（`+3.319%`），超过“不高于 baseline +1%”的次级要求。
- **深水和 bias 是否改善？否。** deep MAE 增加 `0.032538 m`（`+4.650%`）；整体 bias 更负 `0.026191 m`，deep bias 也更负。
- **是否需要扩展到五种子？否。** 预设触发条件是 MAE 相对差异绝对值 `<0.3%`；实际为 `1.590%`，不属于近似持平。
- **是否可因效率提升而接收？否。** 候选参数、显存和吞吐更优，但它不满足精度替代的 primary/secondary 标准。

因此，历史 V15 无光学指标 `0.40226` 也不被用来放宽门槛：当前严格 Matched V15 的多 seed 均值更强，候选不被接受为替代版本。

## 10. Graph / KAN / weak physics 的分别回答

| 模块 | 审计回答 |
| --- | --- |
| Graph | 有实际贡献：graph-off 使 seed-20260904 的 MAE 增加 `0.003913`，update/input RMS 为 `0.091273`，gate 非退化；Graph-A 被保留。Graph-B 的独立筛选失败，因此未保留 bottleneck 扩展。 |
| KAN spline | 有小但非零的实际贡献：spline-off ΔMAE `+0.000151`，base-off ΔMAE `+0.000460`，spline/base RMS `0.586361`，最终 checkpoint spline gradient norm `0.0085353` 且 finite。它仍是后续优化优先项，而不是可删除模块。 |
| weak physics | 实际生效：`lambda_phys>0`、有资格 pair、三种子 physics gradient 非零，筛选期 Physics-A 略优于 Physics-B/simple-fixed。可是没有可重复的多 seed 净精度改善证据，因此不能宣称该约束提升了最终平均精度。 |

## 11. 关键路径

以下路径均可直接复核；其中 candidate 仅是**被拒绝的研究候选**，并非推荐部署权重。

| 内容 | 路径 |
| --- | --- |
| 最终候选配置（被拒绝） | `configs/pa_hydrokan/subset1000_s1_v15_2_final.xml` |
| 严格 Matched V15 配置 | `configs/pa_hydrokan/subset1000_s1_v15_2_final_matched_v15.xml` |
| Candidate seed-20260904 raw / EMA | `runs/optimization/hydrokan_s1_v15_2/final_candidate/seed_20260904/best_raw.pth` / `best_ema.pth` |
| Matched V15 三种子根目录 | `runs/optimization/hydrokan_s1_v15_2/final_matched_v15/` |
| Candidate 三种子根目录 | `runs/optimization/hydrokan_s1_v15_2/final_candidate/` |
| 筛选汇总 | `artifacts/optimization/hydrokan_s1_v15_2/candidate_summary.csv`、`candidate_summary.json` |
| 最终三种子与决策 | `artifacts/optimization/hydrokan_s1_v15_2/seed_summary.csv`、`final_seed_summary.json`、`final_decision.json` |
| paired bootstrap | `artifacts/optimization/hydrokan_s1_v15_2/paired_bootstrap.json` |
| Graph/KAN 反事实与诊断 | `artifacts/optimization/hydrokan_s1_v15_2/final_kan_counterfactuals/`、`final_kan_diagnostics/`、`final_kan_gradient_probe.json` |
| CUDA profile 与总验证 | `artifacts/optimization/hydrokan_s1_v15_2/final_profile.json`、`final_profile_matched_v15.json`、`verification_summary.json` |
| 本报告 | `docs/HYDROKAN_S1_V15_2_ENGINEERING_REPORT.md` |

## 12. 未完成内容与下一步边界

没有因 CUDA、显存、数据、环境或代码错误而遗漏的已选定最终训练；CUDA/BF16 profile 和三种子训练均完成。

- `KAN grid=6` 未运行，因为其前置条件 KAN-map（Experiment 3A）未获得收益；这是预注册的停止规则。
- 五种子验证未触发，因为相对 MAE 差异为 `1.590%`，不在 `<0.3%` 的近似持平区间。
- 没有把 candidate 标注为部署/精度推荐模型；当前没有被接受的 V15.2 精度替代 checkpoint。
- 若后续开启新一轮研究，应把 signed-grade saturation、种子敏感性和弱物理项与 Matched V15 的交互作为新的 validation-only 假设重新设计；不得用本轮 test split 或单次筛选胜负做调参。
