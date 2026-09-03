# subset1000 模型优化与像元精度报告

## 1. 数据与审计

- 数据根目录：`/home/whu/桌面/myData/Flood_depth/subset1000`
- manifest：997 个样本（train=825、val=89、test=83），所有 11 个栅格组均完整。
- 审计结果：`artifacts/dataset_audit/subset1000_audit.json`，状态为 `ready`。
- 运行 contract：`artifacts/dataset_audit/subset1000_contract.json`。
- train-only 统计：`artifacts/dataset_audit/subset1000_train_stats.json`，未读取 val/test 像元构建归一化或深度分层边界。
- 连续输入：27 个通道；使用固定 `hydro_compact` 语义（S1 VV/VH 及变化、S2 B3/B8/B11 及 NDWI/MNDWI、DSM/坡度），未将事件 ID 或标签派生量送入模型。

原始数据在本轮开始前已更新；此前发现的 train 标签缺失问题已消失。本报告及所有配置均直接引用更新后的原始 subset1000，早先生成的 `artifacts/dataset_views/subset1000_labeled` 视图未参与训练。

## 2. 对比协议

模型选择只使用 train/val，主指标为有效监督像元的 `pixel_micro_mae`；test 仅在最终 checkpoint 固定后评价一次。所有运行使用 seed `20260903`，无事件区分输入。

| 实验 | 配置 | 训练设置 | 最优 val raw epoch | val pixel MAE (m) | val RMSE (m) | val P90 (m) | val bias (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| v13 baseline | `configs/pa_hydrokan/subset1000_v13_corrected.xml` | batch=4, AMP off, 20 epochs | 18 | 0.39802 | 0.74616 | 1.03250 | −0.27986 |
| v13.2 KAN-A | `configs/pa_hydrokan/subset1000_v13_2_kan_a.xml` | batch=8, AMP on, 30 epochs | 27 | 0.40059 | 0.75037 | 1.04658 | −0.29675 |
| v13 final | `configs/pa_hydrokan/subset1000_v13_final.xml` | batch=4, AMP off, 25 epochs | 13 | **0.39584** | **0.74742** | **1.02924** | −0.28000 |

KAN-A 在本数据规模和当前损失权重下没有超过 v13（MAE 高约 0.00475 m），因此没有强行替换像元精度更优的 v13。KAN-A checkpoint 和 val 结果仍保留为可复现实验对照。

## 3. 最终 test 基准

最终 checkpoint：`runs/optimization/hydrov13_2_subset1000/final_v13/best_raw.pth`（由 val epoch 13 固定）。

test 结果：`artifacts/optimization/hydrov13_2_subset1000/final_v13_test_raw.json/summary.json`。

| 指标 | test |
|---|---:|
| pixel_micro_mae | **0.19071 m** |
| pixel_micro_rmse | 0.36143 m |
| pixel_micro_p90_absolute_error | 0.36321 m |
| pixel_micro_bias | +0.00361 m |
| pixel_micro_r2 | 0.05144 |
| within 0.25 m | 0.80336 |
| within 0.50 m | 0.94205 |
| within 1.00 m | 0.97902 |
| evaluated pixels | 583,808 |

test 误差明显低于 val，反映两个 split 的事件/深度分布差异；该结果应作为本 subset1000 的独立 test 基准，不应解释为通过 test 调参获得的改进。

## 4. 复现与工程验证

- 最终画像：`artifacts/optimization/hydrov13_2_subset1000/final_v13_profile.json`。
- 参数量：5,076,979；batch=4 平均前向 0.03698 s（约 108.17 samples/s）。
- forward+backward 峰值显存：15,197,456,384 bytes；梯度有限且无 AMP 溢出。
- test 预测 GeoTIFF/可视化输出位于 `artifacts/optimization/hydrov13_2_subset1000/final_v13_test_raw.json/`。
- 完整测试：106 passed，0 failed，0 skipped（3 个既有兼容性 warning）。

## 5. 结论与后续

本轮在更新后的 subset1000 上完成了数据重审计、train-only 归一化、物理边特征统计、v13/KAN-A 对比、独立最终重训和 test 基准。当前应发布 v13 final 作为像元精度主模型；若后续继续提升 KAN-A，应优先研究图边特征与 DSM/水面高程约束的任务适配及损失权重，而不是泛化的学习率或网络堆深。
