# 新配置系统 `validate_config()` 校验指南

本文从属于 `parser_refactor_plan.md`，只规定新嵌套 YAML 在完成结构规范化、组件参数规范化和路径解析后的语义校验。旧扁平 YAML、旧 `CUSTOM_DICT` 和 `parse.copy` 不属于新接口。

## 1. Validator 的输入前置条件

进入 Validator 前，配置已经满足：

```text
run.mode 与 task.type 已读取并受支持
已选择并组合当前 Mode/Task Schema
未知字段已拒绝
必填字段已补齐或报错
普通默认值已填充
所有 params 已由组件专属规范化器转换为 dict
所有路径字段已转换为绝对 pathlib.Path
CLI Override 已完成规范化和路径解析
```

Validator 只校验，不修改配置。

成功：

```text
返回 None
```

失败：

```text
抛出包含完整字段路径、预期规则、实际值和实际类型的异常
```

## 2. 当前已确认的顶层运行边界

```text
optimization：仅 train 必须；train 时不能是 null/{}``
evaluation：train/test 按实际流程要求；inference 当前不要求
runtime：缺失/null/{} 默认 cpu、index=null、count=1
metrics：当前使用阶段必须是非空列表；type 统一 Registry 解析；params 按 metric type 动态确定；同一阶段不能重复
Task-Metric 兼容性：Parser 不检查，由运行时统一 output_dict 提供输入
Checkpoint：Parser 不选择 best/last，由 Runner 按候选路径查找，全部失败才报错
```

## 3. 调用顺序

```text
validate_experiment()
→ validate_run()
→ validate_task()
→ 按存在性调用组件 Validator
→ validate_mode()
→ validate_task_compatibility()
→ validate_cross_fields()
```

禁止无条件访问当前 Schema 不包含的分区。

## 3. 通用规则

### 3.1 数值类型

严格整数：

```text
type(value) is int
```

数值：

```text
type(value) in (int, float)
```

所有浮点、权重、概率、学习率和 Epsilon 使用 `math.isfinite()` 拒绝 NaN 与 Inf。

### 3.2 字符串

```text
必须是 str
strip 后非空
用于 Registry 查询时使用规范化键
```

建议 Parser 在返回最终配置前已把所有 `type` 转成 Registry 键，Builder 不再重复规范化。

### 3.3 路径

文件：

```text
pathlib.Path
绝对路径
存在
is_file
大小大于 0
```

目录：

```text
pathlib.Path
绝对路径
存在
is_dir
按组件规则决定是否必须非空
```

Checkpoint 还要求 `.pt`；配置快照要求 `.yaml` 或 `.yml`。

## 4. 顶层分区

### `experiment`

```text
experiment.name: 非空字符串
experiment.seed: 严格 int 且 >= 0
```

### `run`

```text
run.mode: train/test/inference
```

并且必须等于入口传入的 `expected_mode`。

Train：

```text
run.resume_from: None 或有效 .pt 文件
```

Test/Inference：

```text
run.checkpoint.path: 有效 .pt 文件
run.checkpoint.train_config: 有效 YAML 文件
```

### `task`

```text
task.type: 当前支持的业务任务键
task.params: 最终必须是 dict
```

当前支持：

```text
mlm_pretrain
qa_sft
```

## 5. 组件校验

### 5.1 Tokenizer

`components.tokenizer.type` 必须受 Tokenizer Registry 支持。

当前公共参数：

```text
vocab_file: 有效非空文本文件
do_lower_case: bool
strip_accents: bool 或 None
max_input_chars_per_word: 正整数
```

词表特殊 Token 和重复 Token 继续由 Tokenizer 构造过程负责。

### 5.2 Model

`components.model.type` 必须受 Model Registry 支持。

Tiny-BERT 参数：

```text
hidden_size > 0
max_position_embeddings > 0
type_vocab_size > 0
num_heads > 0
intermediate_size > 0
num_layers > 0
embedding_dropout、attention_dropout、hidden_dropout ∈ [0,1)
layer_norm_eps > 0
hidden_size % num_heads == 0
```

`components.model.init_from`：

```text
None
或包含 kind/path 的 dict
kind 只允许 backbone/model
path 是有效 .pt 文件
```

### 5.3 Losses

```text
components.losses 是非空 list
每项 name 非空
name casefold 后唯一
type 受 Loss Registry 支持
weight 有限且 >= 0
至少一个 weight > 0
params 最终为 dict
```

CrossEntropyLoss：

```text
ignore_index 是严格 int
MLM 当前必须等于 -100
```

`params: null` 的默认值填充不属于 Validator；在进入本层前已经规范化完成。

### 5.4 Dataset

```text
data.dataset.type 受 Dataset Registry 支持
data.dataset.root 是有效目录
data.dataset.params 是 dict
data.dataset.splits 中的每个路径是有效非空文件
```

Mode 决定所需 Split；Task 决定 Dataset 类型兼容性。

MLM Dataset 参数：

```text
max_seq_length 是严格正整数且 > 2
```

QA 当前不校验 MLM 专属参数。

### 5.5 Collator

```text
data.collator.type 受 Collator Registry 支持
data.collator.params 是 dict
```

MLMCollator：

```text
mlm_prob 是有限数且位于 (0,1)
```

QACollator 当前没有任务参数时最终为 `{}`。

### 5.6 DataLoader

对每个存在的 Split：

```text
batch_size: 正整数
shuffle: bool
num_workers: 非负整数
pin_memory: bool
drop_last: bool
```

Parser/Builder 不再暗中重写 Valid/Test 行为。建议规则：

```text
valid.shuffle == false
test.shuffle == false
valid.drop_last == false
test.drop_last == false
```

这些可作为当前项目硬约束，保证评估不丢样本且顺序稳定。

### 5.7 Optimizer

```text
type 受 Optimizer Registry 支持
params 是 dict
```

AdamW：

```text
lr > 0
weight_decay >= 0
betas 是长度 2 的 list
每个 beta 有限且位于 [0,1)
eps > 0
```

Builder 可在构造 AdamW 时把列表转换成元组。

### 5.8 Scheduler

```text
type 受 Scheduler Registry 支持
interval 当前只允许 step
params 是 dict
```

StepLR：

```text
step_size > 0
gamma 有限且 >= 0
last_epoch 是严格 int 且 >= -1
```

### 5.9 Loop 与 Validation

Loop：

```text
total_steps > 0
accumulation_steps > 0
log_every_steps > 0
```

Validation：

```text
enabled: bool
every_steps > 0
early_stopping.enabled: bool
early_stopping.patience > 0
checkpoint.save_every_rounds > 0
```

当相关功能关闭时，是否允许保留无效子参数由 Schema 决定；第一版建议保持固定结构，但跨字段 Validator 不执行未启用行为。

### 5.10 Metrics

每个 Metric：

```text
type 受 Metric Registry 支持
params 最终是 dict
MetricCollection 输出字段不能冲突
```

当前参数：

```text
TokenAccuracy.ignore_index: 严格 int，MLM 为 -100
exact_match.params: {}
character_f1.params: {}
```

### 5.11 Inference

```text
inference.max_seq_length: 正整数
inference.decoding.type: 受 Decoding Registry 支持
inference.decoding.params: dict
```

Top-k Sampling：

```text
top_k: 正整数
```

### 5.12 Runtime Device

```text
runtime.device.type: cpu/mps/cuda
runtime.device.index: None 或非负严格 int
runtime.device.count: 正整数
```

当前组合：

```text
cpu  → index=None, count=1
mps  → index=None, count=1
cuda → index=None 或非负 int, count=1
```

硬件可用性可由 Runtime Validator 或入口的 Device Builder 检查，但项目内必须统一，不可重复给出不同规则。

## 6. Mode 规则

### Train

必须存在：

```text
components.tokenizer/model/losses
data.dataset/collator/loaders.train
optimization.optimizer/scheduler/loop/validation
evaluation.metrics.train
runtime.device
```

禁止 `run.checkpoint`。

### Test

必须存在：

```text
run.checkpoint
components.losses
data.dataset/collator/loaders.test
evaluation.metrics.test
runtime.device
```

禁止：

```text
optimization
inference
本次配置中的 tokenizer/model
```

### Inference

必须存在：

```text
run.checkpoint
inference
runtime.device
```

禁止：

```text
components
data
optimization
evaluation
```

## 7. Task 兼容规则

MLM：

```text
Tokenizer、Model、Dataset、Collator 类型必须属于 MLM 组合
Dataset params 含 max_seq_length
Collator params 含 mlm_prob
```

QA：

```text
Tokenizer、Model、Dataset、Collator 类型必须属于 QA 组合
不得访问或要求 mlm_prob
```

当前受支持组合不包含：

```text
inference + qa_sft
```

## 8. 跨字段规则

```text
model.hidden_size % model.num_heads == 0

MLM dataset.max_seq_length
<= model.max_position_embeddings

run.resume_from
与 components.model.init_from 互斥

validation.enabled=true
→ valid split、valid loader、valid metrics 必须存在

evaluation.test_after_train=true
→ test split、test loader、test metrics 必须存在

Test/Inference 的 task.type
必须与训练配置和 Checkpoint 对应任务兼容

Test/Inference Model 参数
必须来自训练配置，不允许本次配置覆盖
```

## 9. 错误格式

示例：

```text
config.components.model.params.hidden_size 必须为正整数；当前值：0；当前类型：int
```

跨字段错误：

```text
config.run.resume_from 与 config.components.model.init_from 不能同时设置；当前值分别为：... 与 ...
```

Schema 结构错误、参数结构错误、路径错误和业务语义错误可以使用不同异常子类，但对用户必须保留完整配置路径。

## 10. 完成标准

```text
Validator 不修改配置
不访问旧扁平字段
不访问当前 Mode/Task 不存在的字段
所有本地规则和跨字段规则分层清楚
所有错误可定位到 YAML 路径
所有 params 在进入 Validator 前已经是 dict
Builder 接收到的配置无需再次做合法性修复
```
