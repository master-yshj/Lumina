# Lumina

Lumina 是一个基于 PyTorch 的 NLP/LLM 实验框架，用于组织模型训练、测试、推理和可视化，适合学习训练流程和开展模型改进实验。通过 YAML 配置管理运行参数，将数据处理、对象组装、任务计算与训练流程分开组织。

当前为 **Lumina 第 2 版**。面向新生学习的第 1 版见 [lumina_for_freshmen](https://github.com/master-yshj/lumina_for_freshmen)。

## 框架能力

- YAML 配置解析与校验，支持命令行覆盖参数。
- 训练、测试、交互推理和可视化四种运行入口。
- 梯度累积、验证、早停、学习率调度及 Checkpoint 保存与恢复。
- CPU / CUDA 运行、训练混合精度和单机 CUDA 多卡 DDP。
- 配置快照、日志、指标与图表输出，以及 CUDA 显存记录和 OOM 诊断。

仓库内置了用于演示的模型类和方法，包括小型 BERT、MLM 掩码预测、QA 抽取式问答、相关指标，以及注意力热力图和损失曲面。这些实现用于展示框架的使用流程，演示结果不代表实际任务性能。

## 环境与安装

当前开发环境为 **WSL/Linux、Python 3.11**，使用 Miniconda 管理环境。进入包含 `requirements.txt` 的仓库根目录，执行：

```bash
conda create -n lumina python=3.11 -y
conda activate lumina
python -m pip install -r requirements.txt
python check.py
```

依赖版本见 [requirements.txt](requirements.txt)。使用 CUDA 时，需要安装与本机驱动兼容的 PyTorch 构建；项目当前使用 PyTorch 2.9.1，可参考 [官方安装说明](https://pytorch.org/get-started/previous-versions/)。下文命令均在仓库根目录运行，`start.sh` 需要 Bash 环境。

## Demo

`cfg/demo/` 提供 12 份 YAML，默认使用 **CPU、FP32、单进程**：

| 任务 | 训练 | 测试 | 推理 | 可视化 |
| --- | --- | --- | --- | --- |
| MLM | `mlm_train.yaml` | `mlm_test.yaml` | `mlm_inference.yaml` | `mlm_visualize.yaml` |
| QA | `qa_train.yaml` | `qa_test.yaml` | `qa_inference.yaml` | `qa_visualize.yaml` |
| GPT（未实现） | `gpt_train.yaml` | `gpt_test.yaml` | `gpt_inference.yaml` | `gpt_visualize.yaml` |

演示数据位于 `dataset/demo/`，每个任务 100 条，按训练、验证、测试 **8:1:1** 划分；可视化复用验证集。数据由 AI 生成，仅用于演示。MLM 与 QA 共用 `tokenizer/demo/vocab.txt`，无需下载额外语料即可开始 MLM demo。

**GPT 的实现尚未接入。** 四份 GPT 配置保留了 `???` 占位项，不能直接运行。其他配置中的 `???` 是运行前需要填写的权重或训练快照路径。

## 使用流程

### 1. 启动 MLM 训练

```bash
bash start.sh --check-config
bash start.sh
```

第一条仅检查配置与资源路径，第二条启动训练。`start.sh` 当前通过单进程 `torchrun` 运行 `start_train.py`，使用 `cfg/demo/mlm_train.yaml`。

训练产物保存在 `experiments/Demo_MLM_Train/<时间>/`，包含训练快照 `config/config.yaml`、权重 `checkpoints/best.pt` 和 `checkpoints/last.pt`，以及日志、指标和图表。

### 2. 填写测试、推理和可视化的路径

打开 `mlm_test.yaml`、`mlm_inference.yaml`、`mlm_visualize.yaml`，将各自 `run.checkpoint` 下的两个 `???` 改成同一次 MLM 训练的文件路径。例如：

```yaml
checkpoint:
  path: ../../experiments/Demo_MLM_Train/实际时间目录/checkpoints/best.pt
  train_config: ../../experiments/Demo_MLM_Train/实际时间目录/config/config.yaml
```

这段位于 `run` 内，各文件原有的 `run.mode` 保持不变。`path` 指向权重，`train_config` 指向训练时保存的配置快照，用于还原模型等对象。将 `实际时间目录` 替换为本次运行生成的目录名。

随后修改 `start.sh` 中的入口文件和 `--config`，运行 `bash start.sh` 即可切换流程：

| 流程 | 入口文件 | MLM 配置路径 |
| --- | --- | --- |
| 训练 | `start_train.py` | `cfg/demo/mlm_train.yaml` |
| 测试 | `start_test.py` | `cfg/demo/mlm_test.yaml` |
| 推理 | `start_inference.py` | `cfg/demo/mlm_inference.yaml` |
| 可视化 | `start_visualize.py` | `cfg/demo/mlm_visualize.yaml` |

也可以直接调用对应入口，例如：

```bash
python start_test.py --config cfg/demo/mlm_test.yaml
```

MLM 推理输入包含 `[MASK]` 的文本；QA 推理依次输入上下文和问题。终端会显示模型回答，使用 `Ctrl+C` 或 `Ctrl+D` 退出。测试用于在测试集上计算指标，与交互推理用途不同。

### 3. 运行 QA demo

QA 训练默认演示从 MLM 迁移模型主干。打开 `cfg/demo/qa_train.yaml`，将 `components.model.init_from.path` 填为上述 MLM 的 `best.pt` 路径，再把 `start.sh` 切换到 `start_train.py` 和 `cfg/demo/qa_train.yaml` 后运行。若希望从随机权重训练，将整个 `components.model.init_from` 设为 `null`。

QA 训练产物位于 `experiments/Demo_QA_Train/<时间>/`。训练后，填写三份 `qa_test / qa_inference / qa_visualize` 配置中的 `run.checkpoint`，使用 **QA 自己的权重与训练快照**，再按上表切换入口及对应 QA 配置。

## 添加自己的模块

若只替换模型内部的注意力、前馈网络等子模块，在模型中接入自己的实现即可；输入输出和配置参数不变时，无需增加框架注册项。

以**通过 YAML 选择一个新模型**为例：

1. 在 `models/` 对应目录实现模型，保持现有任务需要的构造参数与 `forward` 输入输出接口；接口改变时，需要同步适配 Task 或 Builder。
2. 在 [模型注册表](models/Transformer/modeling/registry.py) 导入新模型，将规范名称与类加入 `canonical_map`，在 `lookup_map` 中登记该名称及所需别名。
3. 在 [schemas.py](parse/schemas.py) 复用或定义参数校验类，并在 [schema_registry.py](parse/schema_registry.py) 的 `build_component_types()` 中，将新模型规范名称映射到该参数类。
4. 在 [任务注册表](tasks/registry.py) 中，将新模型规范名称加入对应任务的 `config_spec['allowed_models']`；随后在自己的 YAML 中填写 `components.model.type` 和 `params`。
5. 先用对应入口的 `--check-config` 检查配置，再用少量数据实际运行需要的流程，确认输出、损失和权重加载正常。

新增损失、指标或可视化方法时，参考同类实现，接入对应注册表、参数校验及必要的任务适配。新增完整任务还需要实现数据处理、Task、Builder 和所需的推理适配器，再在 `tasks/registry.py` 登记。当前版本不会仅凭新增一个类或注册名称自动完成这些接入。

## 配置与运行注意事项

- 可参考 demo 在 `cfg/` 下按实验建立自己的配置目录。`--config` 的相对路径以工作目录为起点；YAML 内的资源路径以该 YAML 所在目录为起点，数据划分文件则相对于 `data.dataset.root`。移动配置后需调整路径。
- 命令行末尾可用 `字段路径=值` 覆盖参数，例如 `optimization.optimizer.params.lr=0.0005`。所有入口均支持 `--check-config`，但配置检查不等于实际运行通过。
- 单卡 CUDA 使用 `runtime.device.type: cuda`、`index: 0`，保留 `strategy: none`、`backend: null`。训练可按硬件支持选择 `fp32 / fp16 / bf16`；CPU 不支持本项目的 FP16 训练。
- 多卡训练使用 `device.type: cuda`、`device.index: null`、`distributed.strategy: ddp`、`distributed.backend: nccl`，并把 `start.sh` 的 `--nproc_per_node` 改为可见 GPU 数量，至少为 2。当前仅支持单机 CUDA/NCCL DDP 训练。
- 独立测试、推理和可视化目前使用单进程 FP32，支持 CPU 或单卡 CUDA；MPS 尚未接入正式运行流程。
- 恢复训练使用训练入口的 `--resume` 指定 `last.pt`，同时将 `components.model.init_from` 设为 `null`，并保持模型、词表和训练配置兼容。
- 测试、推理和可视化结果保存在对应训练实验下的 `test/`、`inference/`、`visualize/`。仓库不提供训练权重或大型语料。
