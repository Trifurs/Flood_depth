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

所有共享的运行参数位于 `configs/base/base.xml`：设备、随机种子、run tag、训练 epoch、
batch size、workers、AMP、优化器、scheduler、早停与评估设置均不依赖命令行参数。
模型结构参数保留在各自模型 XML 中，数据参数保留在
`configs/base/datasets/flooddepthnet_s1_terrain.xml` 中。

`<runtime>` 是统一入口的控制段：

- `run_tag`：同一 seed 的可追溯结果标签；改 seed 时应同步修改它。
- `train`：可选 resume/init checkpoint、批次上限和显式输出目录。
- `evaluation`：split、checkpoint、weights、预测导出和显式输出目录。

默认正式基线为 AdamW（`2e-4`，weight decay `1e-4`）、10 epoch warm-up cosine
schedule、160 epochs 上限、60 epochs 下限、patience 30、batch size 8、8 workers、
bfloat16 AMP 与梯度裁剪 1.0；PA-HydroKAN 额外使用 EMA（`0.9995`）。需要调整任何
参数时，修改 XML，而不是向命令追加覆盖项。

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

传统方法没有可训练参数；对其运行同一入口会执行 XML 中配置的确定性评估：

```bash
python train.py configs/compare/traditional/fwdet_v2.xml
python train.py configs/compare/traditional/tsa.xml
python train.py configs/compare/traditional/fldepth.xml
```

所有模型的统一评估命令同样为：

```bash
python evaluate.py <model-config.xml>
```

默认 `runtime.evaluation.split` 为 `val`，并自动使用同一 `run_tag` 训练目录中的
`best_raw.pth`。完成验证集选择后，将 XML 的该字段改为 `test`，再运行同一条
`python evaluate.py ...` 命令。传统方法没有 checkpoint；`train.py` 与
`evaluate.py` 对它们都会触发同一确定性评估，因此在同一 `run_tag` 下二选一运行即可。

## 结果目录

在未设置显式 `runtime.*.output` 时，路径由配置稳定推导，且同名输出已存在时会拒绝覆盖：

```text
runs/flooddepthnet_s1_terrain/
├── train/<model-run-name>/<run-tag>/
├── ablation/<variant>/<run-tag>/
├── evaluate/<model-run-name>/<val-or-test>/<run-tag>/
├── comparison/<split>/<run-tag>/
├── inventory/
└── infer/<model>/<sample-timestamp>/
```

消融训练也不需要模型专属命令：

```bash
python train.py configs/ablation/pa_hydrokan_no_topographic_affinity_edge_kan.xml
```

每个训练目录保存解析后的配置、数据指纹、校准状态、checkpoint、参数清单和指标。比较模型
始终直接使用 `valid_depth_mask` 作为固定洪水范围；PA-HydroKAN 不接收洪水范围输入。
