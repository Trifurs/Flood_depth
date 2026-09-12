# FloodDepthNet 完整数据集正式运行

项目固定使用完整 FloodDepthNet v3 重平衡发布版：5,323 个训练样本、258 个验证样本和
254 个测试样本。激活输入契约为 `s1_terrain`，只读取 Sentinel-1 T1/T2/change、S1 QA、
DEM、label 与 masks；S2 路径仅保留在原始发布清单的元数据中，绝不注册或读取。

完整数据的审计契约和 train-only 归一化统计位于
`assets/flooddepthnet_s1_terrain/`。仅当原始发布数据、清单或输入语义改变时，才重新运行：

```bash
python tools/prepare_flooddepthnet_s1_terrain_assets.py
```

## 配置即运行参数

所有共享的运行参数位于 `configs/base/base.xml`：设备、随机种子、运行目录时间格式、训练 epoch、
batch size、workers、AMP、优化器、scheduler、早停与评估设置均不依赖命令行参数。
模型结构参数保留在各自模型 XML 中，数据参数保留在
`configs/base/datasets/flooddepthnet_s1_terrain.xml` 中。

`<runtime>` 是统一入口的控制段：

- `run_id_format`：默认 `%Y%m%d-%H%M%S-%f`，以训练启动时刻生成目录 ID；微秒级后缀确保
  同一 seed 的重复实验不会相互覆盖。
- `train`：可选 resume/init checkpoint、批次上限和显式输出目录。
- `evaluation`：split、checkpoint、`source_run`、weights、预测导出和显式输出目录。未指定
  checkpoint 时会选择该模型最近一个包含 `best_raw.pth` 的完整训练；需要严格复现实验时，
  将 `source_run` 设为对应的时间戳目录名。

默认正式基线为 AdamW（`2e-4`，weight decay `5e-5`）、5 epoch warm-up cosine
schedule、200 epochs 上限、60 epochs 下限、patience 30、batch size 12、8 workers、
bfloat16 AMP 与梯度裁剪 1.0。所有深度学习模型统一用
`balanced_composite_error_m` 选择单一 raw checkpoint，不再为 PA-HydroKAN 单独启用
EMA。事件平衡采样使用 replacement 和 `event_balance_power=0.25`；这些共同条件均位于
base/dataset 配置中。需要调整任何参数时，修改 XML，而不是向命令追加覆盖项。

### 当前工作站的实测硬件配置

在完整训练集与 RTX 5090（32 GB）上，PA-HydroKAN 的 batch size 12 已完成完整 epoch，
峰值训练显存约 28.1 GiB；继续增大 batch 缺少稳定余量。8 个 workers、`prefetch_factor=2`、
`persistent_workers=true` 与非阻塞传输下，数据等待约为 0.004 s/10 batch，远小于约
3.65 s/10 batch 的计算时间，因此增加 workers 或预取数量没有实际收益。

基础配置现采用 `deterministic=false`：真实 batch 微基准在相同 BF16、batch size 12 下为
47.86 samples/s，相比严格 deterministic 的 32.53 samples/s 快约 47%，两种设置均为有限
loss。随机种子仍固定；该模式不承诺逐位重现。若需要严格算法确定性，可将该字段改回 `true`，
并接受明显的吞吐下降。TF32 额外开启的收益不足 1%，因此保持 FP32 高精度策略。

## 统一入口

在已激活项目 Python 环境后，所有可学习模型均使用同一条形式的命令：

```bash
python train.py <model-config.xml>
```

例如：

```bash
python train.py configs/pa_hydrokan.xml
python train.py configs/compare/deep_learning/dlsim_attention_unet.xml
python train.py configs/compare/deep_learning/dlsim_linknet.xml
python train.py configs/compare/deep_learning/unet_depth_regression.xml
python train.py configs/compare/deep_learning/resnet18_depth_regression.xml
python train.py configs/compare/deep_learning/unetplusplus_depth_regression.xml
```

传统方法没有训练阶段，因此 `train.py` 会拒绝传统模型配置。完成深度学习模型训练后，使用
一个目录参数自动发现每个模型最新的完整 checkpoint，并在官方测试集上测试；传统模型默认
同时计算：

```bash
python test.py runs/flooddepthnet_s1_terrain/train
```

需要排除传统模型时使用 `--no-traditional`；需要评估目录下所有重复运行而非每个模型最新一次
运行时使用 `--all-runs`。汇总报告包含深度精度、参数量、checkpoint 体积、同步纯前向延迟、
吞吐率、峰值 GPU 显存及端到端测试时间。

所有模型的统一评估命令同样为：

```bash
python evaluate.py <model-config.xml>
```

默认 `runtime.evaluation.split` 为 `val`，并自动使用最近一次完整训练目录中的
`best_raw.pth`。完成验证集选择后，将 XML 的该字段改为 `test`，再运行同一条
`python evaluate.py ...` 命令。正式的全模型测试与传统模型计算统一由 `test.py` 完成。

完整 event 分组 5 折交叉验证使用：

```bash
python validate_k_fold.py
```

也可把其他 `k` 作为唯一位置参数，例如 `python validate_k_fold.py 10`。脚本把原
train+val+test 的全部样本合并，再按 `source_event_id` 重新分为外层折。每轮一折只作
外层测试，其余四折再按 event 划分训练集与内层早停验证集。任何 event 在同一轮都不会
跨训练/内层验证/外层测试角色，每个样本在五轮中恰好一次进入外层测试。每折独立重算
train-only 归一化和深度校准，并自动训练主模型、全部深度学习对比模型及七个消融组合。最终
直接打印每个外层测试折的指标和跨折均值 ± 样本标准差，同时保存完整 CSV/JSON。
由于原 test 样本也按要求纳入了这个全样本流程，交叉验证之外不再另有一份独立外部测试集；
正式泛化结论应以五个互斥外层测试折的统计为准。
每个 model×fold 在独立子进程中执行并逐项落盘。中断后使用
`python validate_k_fold.py --resume` 自动续跑最新未完成会话；已完成项目不重复，
未完成训练从 last checkpoint 继续。正式交叉验证会检查 CUDA 健康状态，
驱动不可用时拒绝静默回退到 CPU。隔离 worker 的
`MKL_THREADING_LAYER=GNU` 也由 base XML 统一配置，正式执行前会在同样的
子进程环境中预检 NumPy、PyTorch 和 CUDA。
新会话同时固化全部模型的合并后配置、源 manifest、全样本嵌套划分协议和运行时代码 SHA-256，
并在每个任务前复核。当前全样本外层测试协议之前创建的旧会话不允许续跑，以免将不同划分或模型的
fold 结果混入同一统计；
当前正式实验必须先使用不带 `--resume` 的命令创建新会话。

## 结果目录

在未设置显式 `runtime.*.output` 时，路径由配置稳定推导，且同名输出已存在时会拒绝覆盖：

```text
runs/flooddepthnet_s1_terrain/
├── train/<model-run-name>/<started-at>/
├── ablation/<variant>/<started-at>/
├── evaluate/<model-run-name>/<val-or-test>/<source-training-run>/
├── test/<started-at>/
├── cross_validation/k<k>/<started-at>/
├── comparison/<split>/<comparison-id>/
├── inventory/
└── infer/<model>/<sample-timestamp>/
```

消融训练也不需要模型专属命令：

```bash
python train.py configs/ablation/pa_hydrokan_wo_tae_kan.xml
```

每个训练目录保存解析后的配置、数据指纹、校准状态、checkpoint、参数清单和指标。比较模型
始终直接使用 `valid_depth_mask` 作为固定洪水范围；PA-HydroKAN 不接收洪水范围输入。

## 训练监控

默认控制台在每个 epoch 输出 `当前 epoch/总 epoch`、本 epoch 耗时、训练损失、验证指标与最优
值、学习率、早停计数、累计耗时和 ETA。完整逐 epoch 标量同时写入对应运行目录下的
`tensorboard/`；查看命令和指标标签见 [RUN_MONITORING.md](RUN_MONITORING.md)。
