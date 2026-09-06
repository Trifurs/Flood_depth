# PA-HydroKAN Hydro-v13 工程优化报告

本报告只记录主模型、公共训练流程和本机实际运行结果。最终配置为
`configs/pa_hydrokan/subset150_v13_final.xml`，最终 checkpoint 为
`runs/optimization/hydrov13/final_seed_20260831/best.pth`。仓库默认的
`subset150_main.xml` 未被自动切换，Hydro-v12 checkpoint 与历史结果均保留。

## 1. 修改前状态

- Git commit：`4bcd6b3db2168f0a723f130b63ef13e9de143352`；开始时工作区无修改。
- Python 3.10.20；PyTorch 2.11.0+cu130；CUDA build 13.0。
- GPU：NVIDIA GeForce RTX 5090；CUDA 和 BF16 均实际可用。
- 修改前测试：68 passed，0 failed，6.28 s。
- 主配置 SHA256：`5b8525d18d9a967e65fb8df70fa901f060198e5dd3f866a6abef4b4044471fd2`。
- Hydro-v12 baseline：
  `runs/train/pa_hydrokan_subset150_hydrov12_raster_depth_balance_20260901_153904/best.pth`，epoch 33。
- 机器可读记录：`artifacts/optimization/hydrov13/prechange_status.json`。

## 2. Baseline 复核与性能画像

Hydro-v12 checkpoint 在完整 validation 上重新评价，而非抄录历史文档值：

| 项目 | 实测值 |
|---|---:|
| validation pixel MAE | 0.522357491 m |
| validation pixel RMSE | 0.792521721 m |
| validation P90 absolute error | 1.193839884 m |
| validation bias | -0.316015172 m |
| 参数量 | 16,745,085 |
| batch=1 前向 | 0.013110 s |
| batch=4 前向 | 0.021951 s |
| batch=4 throughput | 182.222 samples/s |
| batch=4 前向峰值显存 | 1,053,311,488 bytes |
| batch=4 前向+反向 | 0.637518 s |
| batch=4 前向+反向峰值显存 | 7,103,947,264 bytes |
| 梯度范数 | 3.051066 |
| AMP overflow | 未发生 |

基线模块参数量为：S1 encoder 5,925,920，S2 encoder 5,920,736，terrain
1,958,976，fusion 1,838,376，graph 131,954，decoder 969,024，heads 99。
完整 profile 位于 `artifacts/optimization/hydrov13/baseline_profile.json`。

## 3. 发现并处理的结构与流程问题

1. 连续输入通道原先由位置和固定通道数隐式绑定，无法安全筛选波段，也无法减少 Rasterio I/O。
2. 旧双时相特征的大拼接和常规残差块占据大部分参数；跨模态融合更依赖 reliability，内容自适应不足。
3. 地形梯度没有把像元尺寸和中心差分两端有效性表达为同一约束，边缘填充值可能污染局部导数。
4. 旧 Graph-KAN 的消息归一化、单头表达和混合精度 B-spline 计算稳定性仍可加强。
5. 解码器对传感器 skip 与地形提示缺少显式竞争门控，三个任务头的任务隔离较弱。
6. 梯度累积的最后不足组、EMA、BF16 自动选择、配置驱动 optimizer/scheduler、日志开关、checkpoint 开关和恢复身份约束需要统一处理。
7. 多进程训练的输出目录、训练指标、rank-0 validation、停止状态和清理顺序需要同步。

## 4. BandSpec 与选择性栅格读取

`datasets/band_selection.py` 提供不可变 `BandSpec`：

- 从 contract 的精确 `band_descriptions` 解析名称；缺失、重复名称直接报错；
- 保持 XML 给定顺序，并输出名称、零基索引和通道数；
- incidence angle 可作为独立 conditioning，从其原始 S1 时相读取但不混入影像编码通道；
- dataset 打开栅格后仍核对完整 descriptions，只读取模型选中的 band indexes；
- `resolved_model_bands` 写入 dataset metadata、resolved config、model summary 和 checkpoint 身份；
- evaluate、test、infer 和 profile 复用同一解析；旧配置没有 `model_bands` 时读取全部旧通道。

三个训练候选的精确输入如下。

### full_27

- S1 T1：VV、VH、incidence angle；S1 T2：VV、VH、incidence angle。
- S1 change：VV delta、VH delta、`anomaly_raw`、`anomaly_selection`。
- S2 T1/T2：B2、B3、B4、B8、B11、B12。
- S2 change：NDWI delta、MNDWI delta、`water_change_selection`。
- terrain：DSM elevation、slope。

### hydro_core

- S1 T1/T2：VV、VH；conditioning：两个 incidence angle。
- S1 change：VV delta、VH delta、`anomaly_raw`、`anomaly_selection`。
- S2 T1/T2：B3、B8、B11、B12。
- S2 change：NDWI delta、MNDWI delta、`water_change_selection`。
- terrain：DSM elevation、slope。

### hydro_compact（最终选择）

- S1 T1：`VV_pre_db`, `VH_pre_db`。
- S1 T2：`VV_event_db`, `VH_event_db`。
- S1 change：`VV_delta_db`, `VH_delta_db`, `anomaly_raw`。
- S1 conditioning：`angle_pre_deg`, `angle_event_deg`。
- S2 T1：`B3_pre_reflectance`, `B8_pre_reflectance`, `B11_pre_reflectance`。
- S2 T2：`B3_event_reflectance`, `B8_event_reflectance`, `B11_event_reflectance`。
- S2 change：`NDWI_delta`, `MNDWI_delta`。
- terrain：`elevation_m_DSM`, `slope_deg`。

最终序列化结果见 `artifacts/optimization/hydrov13/selected_band_spec.json`。

## 5. 波段屏蔽与重训练筛选

在 Hydro-v12 epoch-33 checkpoint 的完整 validation 上逐波段和逐组置零。该工具只用于敏感性定位，永久选择由随后同 seed 重训练决定。屏蔽基准 MAE 为 0.522349113 m。

| 被屏蔽项 | MAE 变化（m） | 解释 |
|---|---:|---|
| terrain group | +0.025663 | 地形整体重要 |
| S2 change group | +0.018454 | 光学水体变化整体重要 |
| MNDWI delta | +0.010710 | 保留 |
| water-change selection | +0.010442 | 有敏感性，但 compact 重训练不保留 |
| B12 event | +0.011789 | 单点敏感，但 compact 重训练表明不是必要输入 |
| slope | +0.006982 | 保留 |
| incidence conditioning | +0.005876 | 从影像通道移至 conditioning 后保留 |
| B11 pre | +0.006086 | 保留 |

若屏蔽某些单波段后 MAE 暂时下降，只能说明既有 checkpoint 存在冗余或共适应，不能直接证明该波段应删除。因此执行了三组完整波段筛选：

| 候选 | 完成 epoch | 最优 EMA epoch | EMA MAE | EMA RMSE | EMA P90 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| full_27 | 30/45，early stop | 0 | 0.528123 | 0.766850 | 1.151099 | 不保留 |
| hydro_core | 30/45，early stop | 0 | 0.526160 | 0.770436 | 1.169010 | 不保留 |
| hydro_compact | 45/45 | 44 | **0.506451** | 0.779195 | 1.279540 | 保留 |

三组均使用 seed 20260831、同一 v13 结构、同一 optimizer/loss/evaluator，且没有用 `max_train_batches` 截断正式排名。compact 的 EMA MAE 分别比 full 和 core 低约 4.10% 和 3.75%。其 P90 较弱，因此最终训练继续把 P90 作为明确的次要约束观察。

## 6. Hydro-v13 最终结构

以输入 patch `N×C×256×256` 为例：

1. S1 Gated Cross-State Encoder：T1/T2 共享 temporal branch，change 使用独立 branch；在每层组合 state、signed/absolute delta、交互项、change evidence、有效比例和 incidence conditioning。输出
   `N×32×256²`, `N×64×128²`, `N×128×64²`, `N×192×32²`。
2. S2 使用同型 encoder，但没有 incidence conditioning，输出相同四层尺寸。
3. Terrain v13 从选中地形通道与 DSM 原值在线形成 9-pixel `z_hyd`、相对高程、barrier、局部 relief、按 20 m pixel size 缩放且边界有效的 `dz/dx, dz/dy`，形成同尺寸 terrain pyramid。
4. Content-Aware Fusion 在每层联合 reliability、三模态内容摘要、最近邻有效性、有效比例和共同可用性，计算 masked S1/S2 softmax 与 terrain gate；零初始化 cross gamma 提供可控跨模态交互。
5. `N×192×32×32` bottleneck 先通过 dilation 1/2/4 的轻量 context，再进入 4-head、8-neighbour Terrain Graph-KAN。12 维边描述包含地形差、坡度、barrier、传感器有效比例、模态集中度、时差、latent difference 和方向/距离；gate-sum normalization 处理稀疏有效边，head gamma 零初始化使初始映射接近 identity；B-spline recurrence 强制 FP32。
6. Gated FPN 将 bottleneck 投影到 32 channels，逐级恢复至
   `N×32×64²`, `N×32×128²`, `N×32×256²`；每级显式竞争 sensor skip 与 terrain hint。
7. 1/4、1/2、full-resolution auxiliary depth 分别对应内部权重 0.10、0.20、0.0；full 项为零，避免重复主头。
8. 三个独立 task trunk 输出 conditional positive depth、support logits 和 bounded Laplace scale。主要连续输出是 `softplus(raw_depth)`，最终形状均为 `N×1×256×256`。

参数量 5,076,979，低于 25M 限制，相对 Hydro-v12 减少 69.68%。

## 7. 最终损失与 effective schedule

主深度项保持 `sample_depth_bin`：每个 raster 内按冻结 train-depth strata 计算 Smooth-L1，再平均非空 strata；线性深度 Huber beta 为 0.25 m，log1p 深度 Huber beta 为 0.25，log 项权重 0.5。简写为：

`L = L_depth + w_pu L_nnPU + w_unc L_Laplace + w_grad L_gradient + w_aux L_aux + w_wse L_WSE + 1e-6(L_KAN-mag + L_KAN-smooth)`。

- nnPU：epoch 5 开始，10 epoch 线性升至 0.20。
- uncertainty Laplace NLL：epoch 8 开始，10 epoch 升至 0.10。
- masked horizontal/vertical gradient consistency：只使用相邻正标签对，Huber beta 0.10 m；epoch 5 开始，10 epoch 升至 0.02。
- deep supervision：epoch 0 开始，2 epoch 升至外层权重 1.0；内部 1/4、1/2、full 权重为 0.10、0.20、0.0。masked average 下采样保证无有效目标时返回可微零。
- weak local WSE Laplacian：epoch 5 开始，15 epoch 升至 0.02。
- KAN magnitude/smoothness：epoch 0 起权重 `1e-6`。

所有项在实际训练 CSV 中记录原始值与 effective weight。

## 8. 优化器、精度与训练控制

- AdamW，base LR `3e-4`，weight decay `1e-4`。
- KAN spline coefficients：LR multiplier 0.5，weight decay `1e-6`；norm、bias、gamma 不衰减；heads LR multiplier 1.0。
- cosine-with-warmup：5 epoch warmup，最低 LR `1e-6`，按成功 optimizer step 更新。
- batch size 4，gradient accumulation 1，最后不足组按实际 sample count 归一化，`drop_last=false`，gradient clipping 1.0。
- RTX 5090 上 `amp_dtype=auto` 实际解析为 BF16，不使用 GradScaler；FP16 设备路径保留 GradScaler 与 skipped-step 检测。
- EMA decay 0.995，100-step warmup；best checkpoint 由 EMA validation pixel MAE 选择，同时保存 raw 与 EMA 权重。
- 最少 30 epoch，patience 25，最多 120 epoch，validation interval 1。
- 实测 DataLoader：workers 0/2/4/8 分别为 25.16/43.63/72.10/71.43 samples/s，最终选择 4 workers、persistent workers、prefetch 2。
- checkpoint 保存 semantic identity v2；恢复时固定模型、BandSpec、reliability schema、loss、optimizer、scheduler 和数据语义。旧 v1 identity 仍可严格匹配。legacy checkpoint 的 patience 可从 epoch CSV 恢复。
- DDP：rank 0 生成并广播 run directory；训练 metric numerator/count 全局归约；rank 0 完整 validation 后广播 summary；stop 状态一致；仅 rank 0 写 checkpoint；恢复时派生 rank-specific RNG；cleanup 前 barrier。

## 9. 结构与损失候选的实际短筛选

这些候选只运行 3 epoch，用于工程稳定性和明显劣化筛除，不作为充分收敛排名。

| 候选 | 最优 EMA MAE | 参数 | 峰值显存 | 时间 | 决策 |
|---|---:|---:|---:|---:|---|
| no context | 0.545285 | 4,960,818 | 12,707,101,184 B | 42.31 s | 拒绝；初期明显变差 |
| no graph | 0.522571 | 5,076,979 | 11,944,316,928 B | 39.77 s | 未证明 graph 的独立增益；最终保留开关和任务特定 graph |
| legacy blocks | 0.539830 | 11,504,963 | 9,018,342,912 B | 36.07 s | 拒绝；参数翻倍且初期较差 |
| no deep supervision | 未得到指标 | 5,076,880 | 12,695,502,848 B | 1.71 s | 第一个 epoch 出现非有限训练值 |
| no deep supervision retry | 未得到指标 | 5,076,880 | 12,695,502,848 B | 1.48 s | 独立重试同样失败，拒绝 |
| Huber beta 0.50 | 0.524347 | 5,076,979 | 12,725,620,736 B | 37.92 s | 不优于 beta 0.25 起点，拒绝 |
| no gradient | 0.523372 | 5,076,979 | 12,725,620,736 B | 37.92 s | epoch 0--2 尚未进入 gradient schedule，结果与主配置相同，判为不充分而非增益证据 |

因此可靠结论是：efficient blocks 和 context 保留，legacy blocks 与 beta 0.50 不保留，关闭 deep supervision 的当前组合不稳定；graph 与 gradient 的独立长期增益尚未由短跑充分隔离，二者保持可配置并随最终集成运行。

## 10. 完整集成训练与评价

hydro_compact 的 epoch-44 `last.pth` 原样复制到最终目录，并带 optimizer、scheduler、EMA 和 RNG 继续到 120-epoch 上限；epoch 103 触发 early stopping。总实际训练时间为 554.220560 s + 733.383074 s = 1287.603634 s，正式训练峰值显存 12,766,997,504 bytes。best checkpoint 为 EMA validation MAE 最优的 epoch 78。

在 epoch-78 best checkpoint 上重新完整评价：

| split / weights | pixel MAE | pixel RMSE | P90 | bias | <=0.25 m | <=0.50 m | <=1.00 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation baseline v12 | 0.522357 | 0.792522 | 1.193840 | -0.316015 | 0.422639 | 0.639552 | 0.844657 |
| validation final raw | **0.488193** | **0.762562** | 1.234479 | -0.350461 | 0.468135 | 0.669986 | 0.850411 |
| validation final EMA | 0.496575 | 0.770480 | 1.254167 | -0.363840 | 0.464317 | 0.652785 | 0.847675 |
| test baseline v12 | 0.286643 | 0.411775 | 0.585184 | 0.145427 | — | — | — |
| test final raw | **0.230066** | **0.388982** | **0.497548** | 0.011817 | 0.711358 | 0.900754 | 0.976181 |
| test final EMA | 0.232984 | 0.392217 | 0.497172 | 0.009114 | 0.705068 | 0.900824 | 0.973345 |

相对 v12，final raw validation MAE 改善 6.54%，RMSE 改善 3.78%，但 P90 恶化 3.40%；final raw test MAE、RMSE、P90 分别改善 19.74%、5.53%、14.97%。因此保留 v13，但明确把 validation 尾部误差作为未解决项。虽然 checkpoint 按 EMA 选择，该 checkpoint 的 raw weights 在完整 validation/test 上更好，部署评价推荐 `--weights raw`，同时保留 `--weights ema` 复核能力。

最终完整 profile 使用真实 checkpoint：batch=1 前向 0.029521 s，batch=4 前向 0.039894 s，100.266 samples/s，前向峰值 848,687,616 bytes；batch=4 前向+反向 0.357758 s，峰值 12,633,710,592 bytes；梯度范数 2.721411，无非有限梯度或 AMP overflow。v13 的训练 step 更轻，但 4-head directional Graph-KAN 使纯前向吞吐低于 v12。

## 11. 实际验证

- 最终完整 pytest（真实 CUDA）：87 passed，0 failed；包含 Graph-KAN FP16/BF16 forward/backward。
- CPU real-raster smoke：通过；1 train batch、1 optimizer step、1 validation batch、checkpoint save/load 和 GeoTIFF export。
- GPU BF16 real-raster smoke：通过；2 train batches、2 optimizer steps、validation、checkpoint 和 GeoTIFF；峰值 3,263,201,280 bytes。
- batch=8 BF16 可运行：1 train batch 与完整 smoke 通过，但峰值 24,990,502,912 bytes，最终仍采用 batch=4。
- v12 修复训练流：2 个完整 train epoch 和完整 validation 通过，epoch-0 MAE 0.527608。
- 三个波段候选均完成既定筛选或 early stopping。
- 最终训练完成 104 个记录 epoch，early stop 正常。
- raw/EMA validation 与 test 均完整通过。
- 最终 raw infer 已导出 `predicted_depth_m.tif` 及相关 GeoTIFF/可视化产品。
- 所有实测 profile 输出梯度有限且未检测到 AMP overflow。

新增测试覆盖：任意名称选择、错误/重复名称、顺序、legacy fallback、core/compact shape、conditioning 非固定索引、reliability 非 magic index、modality dropout 一致性、terrain 无效边和 pixel-size gradient、Graph-KAN identity/空边/FP32/FP16/BF16、masked auxiliary downsample/可微零、loss schedule、最后不足 accumulation、EMA checkpoint、AMP dtype、optimizer/scheduler/runtime/checkpoint 配置消费、training identity 拒绝和 legacy patience 恢复。

## 12. 新增与修改文件

新增核心代码：

- `datasets/band_selection.py`
- `models/efficient_blocks.py`
- `models/terrain_features_v13.py`
- `models/fusion_v13.py`
- `models/terrain_graph_kan_v13.py`
- `models/decoder_v13.py`
- `models/pa_hydrokan_v13.py`
- `losses/multiscale_losses.py`
- `utils/amp.py`, `utils/ema.py`, `utils/optim.py`
- `tools/analyze_band_importance.py`, `tools/profile_model.py`, `tools/profile_dataloader.py`

新增配置：`subset150_v13_common/full/core/compact/final.xml` 及
`configs/pa_hydrokan/screens/` 中 6 个筛选配置。

修改：`datasets/flooddepth_dataset.py`, `datasets/transforms.py`,
`models/kan_layers.py`, `models/terrain_graph_kan.py`, `losses/depth_losses.py`,
`losses/composite_loss.py`, `tools/train.py`, `tools/evaluate.py`, `tools/test.py`,
`tools/infer.py`, `tools/smoke_test.py`, `utils/checkpoint.py`,
`utils/distributed.py`, `utils/registry.py`, `README.md`。

新增 11 个测试文件：band selection、v13 forward、graph KAN、multiscale loss、loss schedule、gradient accumulation、EMA/checkpoint、AMP、training config consumption、terrain v13、modality dropout v13。

## 13. 机器可读产物

- `prechange_status.json`
- `baseline_profile.json`
- `band_mask_importance.csv/json`
- `candidate_summary.csv/json`
- `dataloader_profile.json`
- `final_model_profile.json`
- `selected_band_spec.json`
- `final_decision.json`

均位于 `artifacts/optimization/hydrov13/`。运行目录位于
`runs/optimization/hydrov13/`，历史目录未覆盖。

## 14. 完整复现命令

```bash
cd "/home/whu/桌面/myCode/Flood_depth"

conda run --no-capture-output -n flood-depth python -m pytest -q

conda run --no-capture-output -n flood-depth \
  python tools/smoke_test.py \
  --config configs/pa_hydrokan/subset150_v13_core.xml \
  --device cpu --batch-size 1 --train-batches 1 --val-batches 1 \
  --output-root runs/optimization/hydrov13/cpu_smoke_reproduction

conda run --no-capture-output -n flood-depth \
  python tools/smoke_test.py \
  --config configs/pa_hydrokan/subset150_v13_core.xml \
  --device auto --batch-size 1 --train-batches 2 --val-batches 1 \
  --output-root runs/optimization/hydrov13/gpu_bf16_smoke_reproduction

conda run --no-capture-output -n flood-depth \
  python tools/analyze_band_importance.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --checkpoint runs/train/pa_hydrokan_subset150_hydrov12_raster_depth_balance_20260901_153904/best.pth \
  --split val --device auto \
  --output artifacts/optimization/hydrov13/band_mask_importance

conda run --no-capture-output -n flood-depth \
  python tools/profile_dataloader.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml --batches 20 \
  --output artifacts/optimization/hydrov13/dataloader_profile.json

conda run --no-capture-output -n flood-depth \
  python tools/train.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --device auto --seed 20260831 \
  --output runs/optimization/hydrov13/final_seed_20260831_reproduction

conda run --no-capture-output -n flood-depth \
  python tools/train.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml --device auto \
  --resume runs/optimization/hydrov13/final_seed_20260831_reproduction/last.pth

conda run --no-capture-output -n flood-depth \
  python tools/evaluate.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --checkpoint runs/optimization/hydrov13/final_seed_20260831/best.pth \
  --weights raw --split val --device auto \
  --output runs/optimization/hydrov13/final_validation_raw_reproduction

conda run --no-capture-output -n flood-depth \
  python tools/evaluate.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --checkpoint runs/optimization/hydrov13/final_seed_20260831/best.pth \
  --weights ema --split val --device auto \
  --output runs/optimization/hydrov13/final_validation_ema_reproduction

conda run --no-capture-output -n flood-depth \
  python tools/test.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --checkpoint runs/optimization/hydrov13/final_seed_20260831/best.pth \
  --weights raw --split test --device auto \
  --output runs/optimization/hydrov13/final_test_raw_reproduction

conda run --no-capture-output -n flood-depth \
  python tools/profile_model.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --checkpoint runs/optimization/hydrov13/final_seed_20260831/best.pth \
  --weights raw --device auto --batch-size 4 --iterations 50 \
  --output artifacts/optimization/hydrov13/final_model_profile.json
```

## 15. 未完成与仍存在的工程问题

1. 第二个正式 seed 20260901 未运行；只有任务要求的 seed 20260831 完整训练。因 validation MAE 改善超过 0.5%，本轮没有追加约 20 分钟的第二次正式训练。
2. `no_graph` 与 `no_gradient` 没有完成独立长期收敛实验；短筛选只能确认可运行或指出不稳定，不能量化它们的独立长期贡献。
3. validation P90 比 Hydro-v12 高 3.40%，说明少数高误差像元仍需后续专门处理；本轮没有用通用超参搜索掩盖这一取舍。
4. DDP 控制流已修正并有单进程/辅助函数覆盖，但本机只有一张可见 GPU，未实际执行多 GPU `torchrun` 集成测试。
5. 正式训练是在 `num_workers=0` 下完成；训练后真实栅格 profile 选择了 4 workers。该变更只影响运行时 I/O，不改变模型、损失或 checkpoint 语义，但没有为此重新训练相同模型。
6. 无 deep-supervision 候选连续两次在首 epoch 出现非有限训练值；失败产物已保留用于诊断，没有继续扩大该候选搜索。
