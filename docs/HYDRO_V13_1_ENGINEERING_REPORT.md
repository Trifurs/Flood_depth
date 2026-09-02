# Hydro-v13.1 工程化训练报告

## 1. 范围与约束

- 本轮只新增 `runs/optimization/hydrov13_1` 与对应 artifacts。
- 旧 Hydro-v13 输出保持不变，仅作为 validation reference 记录。
- 模型输入固定为 `hydro_compact` BandSpec；本轮没有进行波段搜索。
- 训练选择指标为像元级 `pixel_micro_mae`，验证阶段只使用 validation split。
- 所有 checkpoint 均记录 raw 与 EMA（若 EMA 开启）状态；最终部署建议使用 raw winner。
- 数据源栅格与审计产物未修改。

## 2. 训练器修正

- 新增 `BalancedRemainderBatchSampler`，将最后的余数分配到各 batch，batch size 最大/最小差不超过 1，避免 singleton。
- 梯度累积按真实样本数归一化，最后一个不完整累积窗口也执行更新。
- checkpoint 保存 `global_step`、sampler、batch sampler、epoch、梯度累积、warmup 与总 optimizer steps。
- resume identity 升级为 v3，包含完整 training context；旧 v1/v2 checkpoint 仍保留兼容读取路径。
- `--resume` 与 `--init-checkpoint` 互斥。
- `--init-checkpoint` 建立新 run，重新初始化 optimizer、scheduler、GradScaler 与 EMA；可选 `--init-weights raw|ema`。
- 训练阶段加入 loss/component finite 检查；异常样本写入 `nonfinite_losses.jsonl` 后中止。
- raw 与 EMA 最优模型独立保存为 `best_raw.pth` 与 `best_ema.pth`，`best.pth` 保留兼容语义。
- `samples_per_second` 改为 logging interval 聚合值，同时记录 data/compute time。
- step 日志增加 graph gate/gamma、fusion entropy、terrain gate、双传感器有效率、不确定性 P90、AMP scale 与显存。
- `training.non_blocking` 传递到递归 tensor 搬运函数。

## 3. 数据与 BandSpec 修正

- 配置选择时校验 S1/S2 T1-T2 数量、语义与顺序一致性。
- legacy full-band 模式显式使用空 `s1_conditioning`，避免隐式读取条件波段。
- 每个连续分支输出有效比例：S1/S2 的 T1、T2、change 与 DEM。
- 稳定的 sensor availability 与 selected-band raster validity 解耦；选中波段的有效比例只作为分支门控输入。
- `s1_available`、`s2_available`、`dem_available` 作为显式 validity 别名提供。
- `prepare_model_inputs` 仅返回白名单输入，并附带六个 branch-validity fraction 张量。

## 4. 增强与缺失传感器处理

- 原 `modality_dropout_probability` 保留兼容映射并发出一次弃用警告。
- 新增独立 `feature_dropout_probability` 与 `sensor_missing_simulation_probability`。
- feature dropout 只清零对应特征，保留语义 availability。
- sensor-missing simulation 同步清零特征/QA、稳定有效性、branch fraction 与 reliability，并重算 `output_valid`。
- 几何增强对影像、terrain、label、mask、reliability 同步作用。

## 5. Hydro-v13.1 结构

- 注册模型名：`pa_hydrokan_v13_1`。
- 四级 encoder 通道固定为 `[32, 64, 128, 192]`。
- `GatedCrossStateEncoder` 对 T1、T2、change 分支分别使用有效比例门控。
- `ContentAwareFusionPyramidV131` 使用独立 S1/S2/terrain 投影，masked sensor softmax 与 terrain gate。
- `GatedFPNDecoderV131` 使用 `[64, 48, 32]` 解码宽度，sensor/terrain 双分支 gate 与 shallow change evidence。
- `VectorizedTerrainGraphKANV131` 支持 2/4 heads，八邻域一次性构造边特征并向量化 KAN 调用。
- Graph-KAN 边特征包括 signed/absolute dz、真实图距离坡度、barrier、sensor/DEM fraction、modality concentration、日期差、latent difference 与邻域距离。
- Graph message 使用 gate-weighted mean × confidence，`gamma` 零初始化保持 identity。
- Graph-KAN 提供 reference direction-loop 路径用于单元测试。
- 参数量为 5,228,915，低于 7M 目标。

## 6. 损失实现

- 有效权重按 epoch schedule 计算，并写入 objective metrics。
- 当 PU、uncertainty、gradient、auxiliary、WSE 或 KAN 的有效权重为 0 时，不调用对应损失函数，仅返回可微零值。
- auxiliary supervision 使用 coarse valid fraction 加权，避免无效 coarse cell 改变尺度。
- 保留像元优先的监督 reduction；训练和评价均输出 pixel micro 指标。

## 7. 配置与运行

- 候选配置：`configs/pa_hydrokan/subset150_v13_corrected.xml`。
- v13.1 候选配置：`configs/pa_hydrokan/subset150_v13_1.xml`。
- v13.1 长程配置保留：`configs/pa_hydrokan/subset150_v13_1_final.xml`。
- validation 选出的校正 v13 长程配置：`configs/pa_hydrokan/subset150_v13_corrected_final.xml`。
- 所有候选使用 seed `20260831`、hydro_compact、batch size 4、余数平衡 sampler。
- 两个候选均从 scratch 训练，未从旧 45 epoch run 续训。
- 最终 winner 从 scratch 启动 120 epoch 上限，`minimum_epochs=40`，满足 patience 后早停。

## 8. Validation 结果

| run | weights | epoch | pixel MAE | pixel RMSE | pixel P90 AE | pixel bias |
|---|---:|---:|---:|---:|---:|---:|
| baseline v13 recheck | raw | 78 | 0.488272 | 0.762605 | 1.234224 | -0.350882 |
| baseline v13 recheck | EMA | 78 | 0.496708 | 0.770577 | 1.254061 | -0.364223 |
| corrected v13 candidate | raw | 30 | 0.496229 | 0.743808 | 1.173010 | -0.223310 |
| corrected v13 candidate | EMA | 39 | 0.516728 | 0.790898 | 1.278043 | -0.345846 |
| v13.1 candidate | raw | 30 | 0.511251 | 0.782042 | 1.245549 | -0.302712 |
| v13.1 candidate | EMA | 27 | 0.521466 | 0.796176 | 1.291102 | -0.331390 |
| final corrected v13 | raw | 36 | 0.473448 | 0.717586 | 1.090896 | -0.198250 |
| final corrected v13 | EMA | 40 | 0.504213 | 0.778839 | 1.278673 | -0.333411 |

候选 raw validation pixel MAE 选择了校正 v13 家族；最终从 scratch 长程训练后 raw checkpoint 达到 0.473448。EMA checkpoint 保留用于复核，但本轮像元 MAE 不优于 raw。

## 9. 资源与 profiling

| model | parameters | forward samples/s | forward+backward peak |
|---|---:|---:|---:|
| baseline v13 | 5,076,979 | 100.31 | 12,652,117,504 B |
| corrected v13 | 5,076,979 | 101.63 | 12,652,117,504 B |
| v13.1 | 5,228,915 | 118.41 | 12,434,391,040 B |
| final corrected v13 | 5,076,979 | 99.78 | 12,652,117,504 B |

- profiling 使用 RTX 5090、CUDA AMP 自动选择 bfloat16、batch size 4。
- baseline profile 同时记录 batch-1 forward 0.03014 s 与 batch-4 forward 0.03988 s。
- final raw forward/backward profile 梯度有限且无 AMP overflow。
- 现有 v12 checkpoint 与旧 v13 checkpoint 均已在当前代码中 strict load 验证通过。
- final 训练墙钟时间约 494.56 s，停止于 epoch 65（0-indexed；运行了 66 个 epoch）。

## 10. 验证与测试记录

- prechange pytest：85 passed，2 skipped。
- 当前普通 sandbox 单元测试：90 passed，2 skipped；提升权限 CUDA 完整测试：92 passed，0 skipped（含 remainder sampler、Graph-KAN 向量化/reference、BandSpec temporal pair 与有效权重跳过测试）。
- v13.1 CPU real-raster smoke：通过，包含前向、反向、checkpoint reload、validation 与 GeoTIFF 导出。
- v13.1 CUDA real-raster smoke：通过，RTX 5090、bfloat16 AMP、checkpoint reload 与 GeoTIFF 导出均通过。
- smoke 产物分别位于 `runs/optimization/hydrov13_1/smoke_cpu` 与 `runs/optimization/hydrov13_1/smoke_gpu`。
- v13.1 validation 样本 infer smoke：通过，GeoTIFF 写出于 `runs/optimization/hydrov13_1/infer_v13_1_smoke`。
- Graph-KAN zero-validity、zero-gamma 与 CPU autocast 路径均有测试覆盖。
- 当前会话普通 sandbox 不暴露 GPU；CUDA 训练/smoke/profile 使用已授权提升权限执行。

## 11. 产物索引

- 候选汇总：`artifacts/optimization/hydrov13_1/candidate_summary.csv`。
- baseline profile：`artifacts/optimization/hydrov13_1/baseline_profile.json`。
- 候选 JSON：`artifacts/optimization/hydrov13_1/candidate_summary.json`。
- 最终决策：`artifacts/optimization/hydrov13_1/final_decision.json`。
- 验证汇总：`artifacts/optimization/hydrov13_1/verification_summary.json`。
- 最终 profile：`artifacts/optimization/hydrov13_1/final_profile.json`。
- 最终 raw checkpoint：`runs/optimization/hydrov13_1/final/best_raw.pth`。
- 最终 EMA checkpoint：`runs/optimization/hydrov13_1/final/best_ema.pth`。
- 最终验证 raw：`runs/optimization/hydrov13_1/final_validation_raw/summary.json`。
- 最终验证 EMA：`runs/optimization/hydrov13_1/final_validation_ema/summary.json`。
- 候选 raw/EMA 验证摘要分别位于两个 candidate validation 目录。
- 训练 step 日志：`runs/optimization/hydrov13_1/final/train_steps.csv`。
- 训练 epoch 日志：`runs/optimization/hydrov13_1/final/metrics_by_epoch.csv`。

## 12. 使用建议

- 当前 validation 结果支持部署 `best_raw.pth`，并按 `subset150_v13_corrected_final.xml` 解析模型。
- 需要继续实验时，应使用新的 run 目录和新的 training identity，不覆盖 final 目录。
- `best_ema.pth` 仅作为保留的平滑权重候选，除非后续 validation 目标改变。
- 第二随机种子和真实多 GPU DDP 本轮未运行；现有实现已保留 DDP sampler/context 接口。
