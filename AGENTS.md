# NLP 教学训练框架 AI 对接文档

> **回答与行动规范最高优先级：** 开始任何工作前必须先阅读根目录 `AI回答方式规范.md`。回答结构、“下一步”的解释、磁盘事实、代码审查顺序和代码修改授权以该规范为唯一标准；本文仅补充项目技术约定，冲突时无条件服从该规范。
>
> **代码修改唯一授权关键词：** 只有用户在当前请求中完整、连续、原样说出“直接修改代码完成任务”，并明确代码任务范围，才允许修改代码；任何其他表达都不能授权代码编辑。文档、配置、数据及运行操作仍需分别明确授权。

## 1. 文档用途

本文档面向后续接手本项目的 AI。开始任何开发、审查或答疑前，必须先阅读本文档，并以磁盘中的最新代码为准。

项目已经跨过最早期的最小 MLM 骨架阶段，当前磁盘包含 MLM、部分 QA、训练基础设施和分层 Metrics；最新状态是完成两个通用文本指标及 Registry 后暂停开发。用户希望亲自编写代码，AI 的主要职责是：先完整审查，再解释下一步目标、接口、张量形状、职责边界和检查标准；除非用户明确索取，否则不要直接给出完整实现代码；没有当前请求中的代码修改安全词和明确范围时，绝不编辑代码。

### 1.1 最高优先级回答与授权协议

完整协议见根目录 `AI回答方式规范.md`，本文不得另行解释或降低标准：

> **代码修改安全词只有一个：** `直接修改代码完成任务`。只有用户在**当前请求**中完整、连续、原样说出这十个字，并同时明确代码任务和代码范围，才允许修改代码。历史消息、文档引用、AI 复述、缺字、换词、近义表达，以及“下一步”“继续”“实现”“帮我改”“你直接做”等都不构成授权。

- “下一步”可能表示当前步骤没讲清楚。必须先检查磁盘实现和此前说明是否完整，不得机械推进。
- 说明实现任务时，必须给出完整执行流程、直接数据流、所需库/函数/方法及其用法、完成标准和复审后的真正下一步；默认不直接给完整代码或伪代码。
- 用户表示写好后，必须重新读取磁盘，并按“结论、必须修改、建议修改、已正确、数据流与参数流、当前完成标准、下一步唯一最小动作”的顺序回答。存在必须修改项时停留当前步骤。
- 文档、配置、数据、训练、测试、推理、依赖安装及其他命令需要分别明确授权；代码安全词不会自动授权这些操作，任何授权都不得扩大范围。
- 用户只授权文档修改时，不得顺带修改代码、配置、数据或运行任务。

## 2. 强制回答方式

- 默认使用中文回答。
- 不要直接给完整代码；优先说明思路、执行顺序、输入输出、形状变化、错误原因和修改方向。
- 用户提交代码后必须重新读取磁盘，并按“结论、必须修改、建议修改、已正确、数据流与参数流、当前完成标准、下一步唯一最小动作”的顺序回答。
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

项目是用于教学的 NLP 训练框架，不是生产级通用平台。当前正式端到端目标链路为：

```text
中文维基语料
→ WordPiece Tokenizer
→ 动态 MLM Mask
→ Tiny-BERT MLM 预训练
→ Backbone 权重迁移
→ CMRC2018 抽取式 QA 微调
→ QA Test / Inference
→ Span 还原原文答案
→ EM / F1
```

当前不做 UI、Dashboard、Agent、RAG、多模态、RLHF、GPT/SFT、自回归生成、多机、FSDP 或 DeepSpeed。CPU、单卡和后续单机多卡 DDP 属于第一版目标能力，但实现顺序必须服从主链依赖。

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

## 6. 当前实现进度（2026-08-26，以当前磁盘和最新交接为准）

当前已经静态具备：

- 新 YAML Schema、Parser、路径处理、配置校验和训练/测试/推理入口；
- WordPiece Tokenizer、QA Offset Mapping、MLM Dataset/Collator；
- Tiny-BERT Encoder、MLM Head/Model、QA Head/Model；
- CrossEntropyLoss、WeightedLoss、AdamW、StepLR；
- MLM Task、Builder、Test 和 Inference 相关模块；
- QA Dataset/Collator、QATask、QA Builder、Backbone 初始化与 Resume 互斥配置语义；
- Trainer 梯度累积、Validation、Early Stopping、训练进度、Logger 和 Checkpoint 接线；
- 实验目录和配置快照；
- Metrics 的 `core/implementations/output` 分层；
- `TokenAccuracy`、通用 `ExactMatch`、通用 `CharacterF1` 及 Registry 注册。

最新完成内容：

```text
Metrics 目录分层
→ ExactMatch
→ CharacterF1
→ Metrics Registry
```

`ExactMatch` 和 `CharacterF1` 的通用输入契约为：

```text
predictions: list[str]
references: list[list[str]]
```

它们不负责 QA Span 解码，不读取 `start_logits`、`end_logits`、`offset_mapping` 或 `context`。QA 任务适配层未来应先生成标准化的 `predictions/references`。

当前验证状态：

- 最新 Metrics 改造只完成源码和静态审查，没有单元测试或组合运行证据；
- 当前 Trainer、Checkpoint/Resume、QA 和 Metrics 没有统一的最新端到端运行证据；
- 静态审查、Linter 和运行验证必须分别报告；
- 未真实运行的能力不得称为“已经运行通过”。

## 7. 当前暂停状态与唯一后续方向

用户当前要求暂停后续开发。不得自动继续修改代码、运行测试、训练或推理。

如果用户后续重新要求推进，默认先做：

```text
重新读取 Metrics 文件和所有导入方
→ 核对 Registry 实际键名
→ 设计 predictions/references 指标上下文
→ 输出静态影响分析
→ 等待当前请求中的代码修改安全词和明确代码范围
```

当前不要直接进入：

- Trainer 大规模指标接入；
- QA Span 解码；
- QA Test / Inference；
- 长文本滑动窗口；
- 端到端 EM/F1；
- 未经单独授权的训练、测试或推理运行。

## 8. 当前配置结构与参数所有权

当前顶层配置以 `parse/parser.py` 的 `CUSTOM_DICT` 和当前 YAML 为准，主要包含：

```text
name
seed
task
model
tokenizer
dataset
dataloader
losses
optimizer
scheduler
train
valid
test
cuda
metrics
```

参数由真正消费它的模块所有：Tokenizer 消费分词参数；Dataset/Collator 消费数据和 Mask 参数；Model 消费网络结构参数；Loss、Optimizer、Scheduler 消费各自参数；Trainer 消费训练、验证和梯度累积参数；入口管理实验名称、种子、路径和 Builder 选择。不得根据本文件中的旧设计猜测接口，必须读取当前 Parser、YAML、Registry、构造函数和调用方。

Tokenizer 派生参数仍不应在 YAML 中重复声明：

- `len(tokenizer)` → 模型 `vocab_size`；
- `tokenizer.pad_token_id` → 模型 `padding_idx`；
- 特殊 Token ID → Collator。

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