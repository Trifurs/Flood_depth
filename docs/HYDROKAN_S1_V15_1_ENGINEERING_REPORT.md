# PA-HydroKAN-S1-V15.1 工程与受控实验报告

## 范围与基线

本次工作只使用 Sentinel-1、地形和许可的 S1 QA 特征；模型、训练和验证流程均不打开 Sentinel-2 文件。所有正式结果仅来自 `val` split，监督域统一为 `valid_depth_mask & output_valid`，共 732,718 个正深度像元。没有读取或报告 test split。

历史无光学 V15 的 canonical 验证基线为 `best_raw.pth` 的 zero-based epoch 35：MAE 0.402263、RMSE 0.765960、P90 1.056909、bias -0.305506。该历史 run 实际写到了 epoch 60，但此 checkpoint 在 epoch 35 被选中；因此 V15.1 的正式准入采用同为 epoch 35 的受控比较。为避免数值复算误差，要求至少 0.5% 的 MAE 实质改善，即 MAE 不高于 0.400252；后续 epoch 的最低值不能绕过该门槛。

候选间统一使用 seed=20260904、同一数据划分、batch size=8、gradient accumulation=2、AdamW、学习率 2.5e-4、45-epoch 上限、canonical validation mask、raw/EMA 双评价。历史 V15 的 seed 为 20260903、AMP 为 FP16；这部分无法回溯改写，故采用严格的同 checkpoint epoch 门槛，并且所有新候选均明显未达门槛。

## P0 修复

- 新增 `datasets/supervision_masks.py`，训练、loss、validation、checkpoint monitor 统一使用 canonical positive mask。
- `minimum_event_band_fraction=1` 明确要求事件期 S1 观测支持；`output_valid` 由统一规则产生。
- internal change 仅在 pre/event pair 有效时计算；external change 由其自身有效性控制；两路经可解释 mixer 融合，无证据时输出零变化且不产生 NaN。
- 图网络的实际特征尺度与配置对齐：corrected 为 stride 8（160 m 节点间距）；KAN/simple 为真实 stride 4（80 m 节点间距）。
- KAN descriptor 只进行一次 robust tanh mapping；edge 统计量只从训练集构建并以 SHA-256 固定到 config、checkpoint fingerprint 和图 identity。
- 修复 context、decoder widths、terrain 初值、support off、depth 初值和配置消费路径；旧 V15 checkpoint 仍可加载。

## V15.1 结构

`corrected` 是仅 P0 修复的 V15 对照；`KAN` 仅替换图/KAN 路径；`simple` 在 KAN 路径基础上使用精简的事件优先共享 SAR encoder、单一 terrain residual、2 个安全 S1 QA（event observation count、selected event day offset）和无 90° 旋转增强。

- 输入：S1 pre/event VV/VH、变化特征、入射角、DSM/slope 和 S1 QA；无 S2 key。
- change mixer：pair-valid internal change 与 external change 分支分别门控；pre context 单独注入。
- terrain：ground proxy/terrain residual 受有界 alpha 与 terrain gate 控制，避免重复 reliability residual。
- 图/KAN：四个命名 descriptor（signed grade、absolute grade、barrier magnitude、local surface complexity），grid=4、heads=2、图通道=64、stride=4；不含 signed static direction prior。
- decoder 与输出：多尺度 decoder、conditional-positive depth 和 uncertainty scale；support head 默认关闭。

## KAN 有效性

V15.1-KAN raw checkpoint 的 validation-only 诊断位于 `artifacts/optimization/hydrokan_s1_v15_1/kan_diagnostics.json`。

- 首个 backward 的 KAN 非零梯度测试通过。
- spline/base RMS ratio=0.5412，非零且不是 base-only 退化。
- graph update/input RMS ratio=0.04375，final graph gate 均值=0.4621，gamma 均值=0.03265。
- 总 knot boundary saturation=0.1118；signed-grade 为 0.1950，其他 descriptor 为 0.0576–0.0979。非负 descriptor 的第一个负区间为空属于定义域结果，其余区间不呈严重单点集中。
- 两个 head 的平均绝对曲线差异分别为 signed grade 0.1553、absolute grade 0.0550、barrier 0.0192、complexity 0.0361，未学成相同曲线。

因此不触发 KAN-B：KAN 路径本身有效，问题是精度未达到历史 V15 的准入线，而不是 KAN 无效或数值退化。

## 实际损失与训练日程

corrected/KAN 实际使用：

`L = L_depth + 0.20 L_log + 0.02 L_bias + 0.02 L_exceed + 0.005 L_gradient + 0.08 L_aux + 0.005 L_unc + 1e-6 L_KAN`。

其中 `L_final`、PU、WSE、tail 均为 0；auxiliary 从 epoch 0 开始、3 epoch warmup，uncertainty 从 epoch 5 开始、5 epoch warmup。depth 初始 bias 对应 0.1 m，uncertainty 初始尺度为 0.35 m。

simple 实际使用 task-adaptive pixel-micro 目标：

`L = L_depth(soft-depth-balance) + 0.05 L_log + 0.04 L_aux + 0.005 L_unc + 1e-6 L_KAN`。

其中 bias、exceedance、PU、gradient、WSE、tail、final 项均为 0；soft depth weights 在训练分布上计算且受 [0.5, 3.0] 约束、均值为 1。auxiliary 常数权重，uncertainty 从 epoch 10 开始、3 epoch warmup。

## 结果与选择

下表均为 raw checkpoint 的完整 validation 复核；深水定义为训练分箱 `[0.5 m, +inf)`。峰值显存为同 batch=8 的 forward/backward 实测值；吞吐为同 batch=8 的实测推理吞吐。

| 模型 | matched epoch 35 MAE | best raw epoch | MAE | RMSE | P90 | bias | 深水 MAE / bias | 参数量 | 峰值显存 | samples/s | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 历史 V15 S1-only | 0.402263 | 35 | 0.402263 | 0.765960 | 1.056909 | -0.305506 | 0.810610 / -0.799954 | 3,533,779 | 19.79 GB | 139.34 | 保留 |
| V15-corrected | 0.421335 | 38 | 0.418143 | 0.778429 | 1.091990 | -0.292340 | 0.825408 / -0.814708 | 3,524,466 | 19.99 GB | 130.88 | 拒绝 |
| V15.1-KAN | 0.420278 | 38 | 0.415427 | 0.767597 | 1.090515 | -0.284201 | 0.814719 / -0.801418 | 3,442,434 | 20.30 GB | 137.80 | 拒绝 |
| V15.1-simple | 0.411780 | 38 | 0.409477 | 0.767078 | 1.042914 | -0.284413 | 0.803777 / -0.790111 | 2,577,918 | 16.71 GB | 167.49 | 拒绝 |

EMA 也单独复核：corrected=0.424721（epoch 44）、KAN=0.423973（epoch 44）、simple=0.415976（epoch 44），均不优于对应 raw checkpoint。

simple 是新候选中最好的：其 P90、bias 和深水指标优于历史 V15，并且更轻、更快；但其主指标 MAE=0.409477，较历史 V15 高 0.007213（约 1.79%），且 matched epoch 35 MAE=0.411780 未通过 0.400252 的实质改善门槛。按照“不超过既有 V15 就不接受”的要求，最终保留历史 V15 S1-only `best_raw.pth`，不启动新的 100-epoch final 训练。

完整机器可读结果见：

- `artifacts/optimization/hydrokan_s1_v15_1/candidate_summary.csv`
- `artifacts/optimization/hydrokan_s1_v15_1/candidate_summary.json`
- `artifacts/optimization/hydrokan_s1_v15_1/final_decision.json`
- `artifacts/optimization/hydrokan_s1_v15_1/final_profile.json`

## 验证

- CPU V15.1 forward/backward、真实栅格 CUDA BF16 optimizer step、raw/EMA checkpoint load、validation-only KAN curve/occupancy 导出均已完成。
- 最新安全测试：`134 passed, 2 skipped, 2 warnings`。
- 命令为 `conda run --no-capture-output -n flood-depth python -m pytest -q --ignore=tests/test_dataset_loading.py`。被排除的 `tests/test_dataset_loading.py` 会显式访问 test split，与本实验的 validation-only 范围冲突。
- 三个候选均进行了全量 validation raw/EMA 复核；未运行 test split 评价或推理。
- 保留的历史 V15 已在 validation 样本 `JRC_2017_244_2017-09-25_OBJ_0056_R000091_C000000` 上成功导出 depth、conditional-depth、uncertainty、support GeoTIFF 与 PNG，路径为 `runs/optimization/hydrokan_s1_v15_1/final_retained_v15_infer_val`。

## 未完成项与复现

未启动 KAN-B，因为 KAN diagnostics 正常；未启动 100-epoch final，因为没有候选获得精度准入；未进行多 seed 重复。若未来需要验证精度改进，应先在同一 seed、同一 45-epoch budget 下跑新的单因素候选，并先通过 epoch-35 MAE 门槛。

复现主要命令：

```bash
conda run --no-capture-output -n flood-depth python tools/build_s1_graph_edge_stats.py
env FLOOD_DEPTH_DISABLE_TQDM=1 conda run --no-capture-output -n flood-depth python tools/train.py --config configs/pa_hydrokan/subset1000_s1_v15_1_corrected.xml --device cuda --output runs/optimization/hydrokan_s1_v15_1/v15_corrected
env FLOOD_DEPTH_DISABLE_TQDM=1 conda run --no-capture-output -n flood-depth python tools/train.py --config configs/pa_hydrokan/subset1000_s1_v15_1_kan.xml --device cuda --output runs/optimization/hydrokan_s1_v15_1/v15_1_kan
env FLOOD_DEPTH_DISABLE_TQDM=1 conda run --no-capture-output -n flood-depth python tools/train.py --config configs/pa_hydrokan/subset1000_s1_v15_1_simple.xml --device cuda --output runs/optimization/hydrokan_s1_v15_1/v15_1_simple
conda run --no-capture-output -n flood-depth python tools/summarize_v15_1_candidates.py
```
