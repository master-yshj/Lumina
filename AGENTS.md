# NLP 教学训练框架 AI 对接文档

> **回答与行动规范最高优先级：** 开始任何工作前必须先阅读根目录 `AI回答方式规范.md`。回答结构、“下一步”的解释、磁盘事实、代码审查顺序和代码修改授权以该规范为唯一标准；本文仅补充项目技术约定，冲突时无条件服从该规范。
>
> **代码修改唯一授权关键词：** 只有用户在当前请求中完整、连续、原样说出“直接修改代码完成任务”，并明确代码任务范围，才允许修改代码；任何其他表达都不能授权代码编辑。文档、配置、数据及运行操作仍需分别明确授权。

## 1. 文档用途

本文档面向后续接手本项目的 AI。开始任何开发、审查或答疑前，必须先阅读本文档，并以磁盘中的最新代码为准。

项目当前处于逐步实现最小 MLM 预训练链路的阶段。用户希望亲自编写代码，AI 的主要职责是：先完整审查，再解释下一步目标、接口、张量形状、职责边界和检查标准；除非用户明确要求，否则不要直接给出完整实现代码，也不要擅自编辑用户正在学习实现的模块。

### 1.1 最高优先级回答与授权协议

完整协议见根目录 `AI回答方式规范.md`，本文不得另行解释或降低标准：

- “下一步”可能表示当前步骤没讲清楚。必须先检查磁盘实现和此前说明是否完整，不得机械推进。
- 说明实现任务时，必须给出完整执行流程、直接数据流、所需库/函数/方法及其用法、完成标准和复审后的真正下一步；默认不直接给完整代码或伪代码。
- 用户表示写好后，必须重新读取磁盘，依次列出“必须修改”“建议修改”“已正确”。存在必须修改项时停留当前步骤。
- 只有当前请求完整、连续、原样包含“直接修改代码完成任务”并明确范围，才允许修改代码；“修改”“实现”“帮我改”“下一步”等其他表达均无效。
- 文档、配置、数据和命令需要分别明确授权；任何授权都不得扩大范围。

## 2. 强制回答方式

- 默认使用中文回答。
- 不要直接给完整代码；优先说明思路、执行顺序、输入输出、形状变化、错误原因和修改方向。
- 用户提交代码后，先指出“已正确”“必须修改”“建议修改”，并解释原因。
- 不要一次规划过多未来功能。每次只推进当前链路中的一个清晰步骤。
- 必须明确告诉用户本步代码应写在哪个现有文件；确需新增文件时，先说明为什么现有文件不能承载该职责。
- 不要把建议说成已经实现。必须明确区分：当前磁盘实现、用户尚未保存的代码、建议方案、未来计划。
- 不要擅自使用 `check.py`、`debug.py` 或 `README.md` 承载训练入口；它们有既定职责。
- 不要因为局部代码看起来合理就立即推进。涉及配置、入口、注册、组件组装或跨模块接口时，必须先完成全链路审查。
- 用户要求“进入下一步”时，先确认上一阶段的代码已保存到磁盘，并检查它对下一阶段的真实依赖。

## 3. 每次工作的强制流程

涉及新增模块、配置修改、训练入口或跨文件改动时，必须执行以下流程：

1. 阅读本文件。
2. 查看完整项目目录，定位相关文件，不能凭文件名猜测内容。
3. 阅读当前任务的上游和下游接口。
4. 阅读配置模板、示例 YAML、注册表和入口相关代码。
5. 确认用户最新代码已保存到磁盘；如果磁盘内容与聊天内容不同，要明确指出。
6. 画出本次修改涉及的数据流和参数流。
7. 列出哪些文件需要修改、哪些文件明确不应修改。
8. 只推进一个最小步骤。
9. 修改完成后检查接口一致性和静态错误。
10. 等整条最小链路组装完成后，再按照用户要求进行统一运行验证。

禁止采用以下工作方式：

- 只读当前文件就设计训练入口；
- 先硬编码组装，之后再考虑现有 YAML；
- 未检查注册表便发明新的类型名；
- 未检查真实构造函数便假设参数；
- 为临时验证挪用已有职责文件；
- 配置中保留不会被代码消费的参数，却声称它已生效；
- 连续增加 Trainer、Scheduler、Checkpoint 等功能而不检查主链路依赖。

## 4. 项目目标

项目是一个用于教学的 NLP 训练框架。当前主要目标是先打通最小 Tiny-BERT MLM 预训练链路，随后再逐步扩展验证、调度器、断点恢复等能力。

当前目标链路：

```text
命令行参数
→ YAML 加载、默认值填充、路径规范化、配置校验
→ Tokenizer
→ MLMDataset
→ MLMCollator
→ DataLoader
→ TinyBertForMaskedLM
→ CrossEntropyLoss
→ MLMTask
→ AdamW
→ Trainer.fit
```

各层职责必须保持清晰：

- Parser：解析命令行、加载 YAML、补默认值、路径绝对化、配置校验。
- Tokenizer：分词、Token/ID 转换、提供词表大小和特殊 Token ID。
- Dataset：读取 JSONL 文本并生成未掩码的 MLM 样本。
- Collator：动态 Padding、动态 MLM 抽样、生成 `labels`。
- Model：输出 `[B, S, V]` 的 MLM logits。
- Loss：接收 `[N, V]` logits 和 `[N]` labels，忽略 `-100`。
- Task：连接 Model 与 Loss，负责 MLM 张量展平。
- Optimizer：根据梯度更新参数。
- Trainer：迁移设备、执行单步训练、按 Iteration 循环和记录 Loss。
- 最终训练入口：消费已校验配置并组装以上对象，不承载组件内部逻辑。

## 5. 当前重要文件与职责

- `parse/parser.py`：命令行和配置处理。
- `cfg/example.yaml`：当前示例配置。
- `models/Transformer/tokenizer/tokenizer.py`：`ExampleTokenizer` 和 `QATokenizer`。
- `loaders/mlm.py`：MLM Dataset、Collator 和文本分块函数。
- `models/Transformer/modeling/modeling.py`：`TinyBertModel` 与 `TinyBertForMaskedLM`。
- `models/Transformer/losses/losses.py`：自定义 `CrossEntropyLoss`。
- `tasks/MLM_PreTrain/mlm.py`：`MLMTask`。
- `models/Transformer/optimizer/optimizer.py`：自定义 `AdamW`。
- `Trainer/trainer.py`：`Trainer`；注意目录名首字母大写。
- `models/**/supported.py`、`tasks/supported.py`、`dataset/**/supported.py`：配置类型白名单。
- `check.py`：环境校验，不是训练入口。
- `debug.py`：调试用途，不是正式训练入口。
- `README.md`：项目和环境说明。

## 6. 当前实现进度

已经实现或基本实现：

- WordPiece 风格 Tokenizer；
- MLM 文本切句、分块、Dataset；
- 动态 Padding 和 80/10/10 MLM Collator；
- Tiny-BERT Encoder 与 MLM Head；
- 输入输出 Embedding 权重共享；
- 自定义交叉熵 Loss；
- `MLMTask`，已将 logits 和 labels 展平后计算 Loss；
- 自定义 AdamW；
- Trainer 的单步训练和按 Iteration 的 `fit()` 骨架。

当前已知问题：

- `Trainer/trainer.py` 中 `fit()` 的日志 Loss 已是 Python 数值，却仍调用 `loss.isfinite()`；应使用已导入的 `math.isfinite`。
- Collator 可能小概率生成整个 Batch 都没有监督位置，进而使平均交叉熵产生 `NaN`；尚未决定在哪一层处理。
- Scheduler、Validation、Test、Checkpoint、Resume、AMP、梯度累积、梯度裁剪和 DDP 尚未接入。
- 正式训练入口尚未创建。
- 当前配置结构与已经实现的 MLM 组件接口不完全一致。

## 7. 当前主要任务：先修改配置链路

用户尚未开始本轮配置修改。下一阶段必须按以下顺序推进，不能跳到训练入口：

1. 修改 `parse/parser.py` 中的 `CUSTOM_DICT`。
2. 修改 `cfg/example.yaml` 与新模板对齐。
3. 修改 `relative_to_absolute_path_full_dict()` 的路径键。
4. 重写 `validate_config()` 中与旧结构绑定的校验。
5. 校验配置字段和组件构造参数一一对应。
6. 修复已记录的 Trainer 有限性检查问题。
7. 创建独立的正式训练入口。
8. 使用命令行组装整条 MLM 链路。
9. 最后统一做最小 Iteration 的前向、反向和参数更新验证。

不能先写入口再补配置，因为最终入口必须由命令行和 YAML 驱动。

## 8. 推荐的新配置结构

目标顶层结构：

```text
name
seed
device
task
tokenizer
dataset
dataloader
model
loss
optimizer
train
```

字段与消费位置：

- `name`：实验名称。
- `seed`：入口设置随机种子。
- `device`：传给 Trainer。
- `task`：通过 `SUPPORTED_TASKS` 校验并选择 Task。
- `tokenizer.*`：构造 `ExampleTokenizer`。
- `dataset.train_path`、`dataset.max_seq_length`：构造 `MLMDataset`。
- `dataset.mlm_prob`：构造 `MLMCollator`。
- `dataloader.*`：构造 PyTorch DataLoader。
- `model.*`：构造 `TinyBertForMaskedLM`。
- `loss.*`：构造 `CrossEntropyLoss`。
- `optimizer.*`：构造 AdamW。
- `train.total_iters`、`train.log_every_iters`：传给 `Trainer.fit()`。

Tokenizer 派生参数不写入 YAML：

- `tokenizer.vocab_size` → 模型 `vocab_size`；
- `tokenizer.pad_token_id` → 模型 `padding_idx`；
- 其他特殊 Token ID → Collator。

当前应移除的旧配置：

- `model.model_path`；
- `modules`；
- `cuda.gpu_nums`；
- 顶层 `dataset_path`；
- 未实现的 `accumulation_steps`；
- 未接入的 `scheduler`；
- 尚未实现的 `valid` 和 `test`。

## 9. 配置必须满足的跨字段约束

- `model.hidden_size % model.num_heads == 0`。
- `dataset.max_seq_length <= model.max_position_embeddings`。
- `dataset.max_seq_length > 2`。
- 当前 `loss.ignore_index == -100`，因为 Collator 仍硬编码 `-100`。
- 三个 Dropout 均满足 `0 <= p < 1`。
- `0 < dataset.mlm_prob < 1`。
- AdamW 的两个 Beta 均满足 `0 <= beta < 1`。
- YAML 中 `betas` 解析为 list，入口构造 AdamW 时转换为 tuple。
- CUDA 字符串格式由配置校验检查，CUDA 是否实际可用由训练入口在运行时检查。

## 10. 张量契约

Collator 输出：

```text
input_ids:      [B, S], torch.long
attention_mask: [B, S], torch.long
token_type_ids: [B, S], torch.long
labels:         [B, S], torch.long
```

模型输出：

```text
logits: [B, S, V], 浮点 Tensor
```

Task 计算 Loss 时：

```text
logits: [B, S, V] → [B×S, V]
labels: [B, S]    → [B×S]
```

Task 返回：

```text
{"logits": [B, S, V], "loss": 标量 Tensor}
```

Trainer 的训练日志返回不应保留计算图或大型 logits，Loss 应通过 `.item()` 转成 Python 数值。

## 11. 验证策略

用户明确要求先完成最小链路，再统一验证。因此在组件逐步实现期间：

- 仍需进行静态审查、形状审查和 linter 检查；
- 不要每完成一个组件就要求用户运行零散测试；
- 等配置、入口和组件组装完成后，再运行少量 Iteration 的 Smoke Test；
- 首次统一验证优先使用 CPU、小 Batch、小序列长度、小模型和极少 Iteration；
- 验证内容至少包括：配置解析、数据加载、动态 Mask、模型前向、有限标量 Loss、反向传播、Optimizer 更新、Trainer 正确停止。

## 12. 后续 AI 开始工作的检查清单

在给出下一步前，必须回答这些问题：

- 用户最新代码是否已保存？
- 本次任务涉及哪些文件？
- 上游输出和下游输入分别是什么？
- YAML 中的每个相关参数由哪段代码消费？
- 是否有字段存在但尚未实现？
- 是否有注册表会拒绝新类型名？
- 是否会破坏 `check.py`、`debug.py` 等既有职责？
- 本步是否真有必要新增文件？
- 当前建议是“必须修改”还是“未来优化”？
- 是否遵守了用户不希望直接获得完整代码的要求？

如果这些问题尚未确认，不要继续指导跨模块实现。