# Parser 重构方案

## 1. 重构目标

当前 Parser 的主要问题不是缺少某个 QA 字段，而是配置系统的职责和层级没有分离：

- `parser.py`、`test_parser.py`、`inference_parser.py` 各自维护一套配置模板和校验逻辑；
- 一个全局 `CUSTOM_DICT` 同时描述 MLM、QA、训练、测试和推理配置；
- `check_and_add_config()` 同时负责递归、默认值、Loss 特殊处理、Metrics 特殊处理和错误路径；
- `validate_config()` 过于庞大，任务差异散落在大量条件分支中；
- `dataset.max_seq_length`、`dataset.mlm_prob` 等字段被错误地当成所有任务共有字段；
- `losses` 和 `metrics` 的 `params` 是多态结构，却被固定模板递归逻辑处理；
- 训练、测试、推理三种运行模式的必需字段不同，但当前模板没有清晰区分；
- Builder 依赖 Parser 返回的字典结构，Parser 重构不能破坏现有 Builder 接口。

重构后的系统应满足：

```text
运行模式决定配置外层结构
任务类型决定任务专属字段
普通字段使用通用递归处理
多态 params 使用组件专属校验
路径解析、结构规范化和语义校验相互独立
训练/测试/推理 Parser 共享底层逻辑
Builder 继续接收字典配置
```

---

## 2. 核心概念：mode、task、component

### 2.1 `mode`

`mode` 表示运行模式：

```text
train
 test
 inference
```

它决定顶层需要哪些运行配置。

### 2.2 `task`

`task` 表示业务任务。建议将“任务”和“运行模式”分开理解。

当前外部任务字符串可以继续兼容：

```text
mlm_pretrain
mlm_pretrain_test
qa_sft
qa_sft_test
```

但 Parser 内部应标准化为：

```text
mlm_pretrain_test
→ mode=test, task=mlm_pretrain

qa_sft_test
→ mode=test, task=qa_sft
```

如果暂时不修改外部配置和任务 Registry，也可以保留原始 `task`，额外生成内部变量：

```text
base_task
mode
```

以后校验统一使用：

```text
mode == "train" and base_task == "qa_sft"
```

不要在各处继续直接判断一长串模式化任务名称。

### 2.3 `component`

组件包括：

```text
model
tokenizer
dataset
dataloader
losses
metrics
optimizer
scheduler
train
valid
test
checkpoint
cuda
```

每个组件的结构校验和语义校验应尽量独立。

---

## 3. 推荐文件结构

建议最终将 `main/parse` 整理为：

```text
main/parse/
├── parser.py                 # 训练入口和兼容导出
├── test_parser.py            # 测试入口和兼容导出
├── inference_parser.py       # 推理入口和兼容导出
├── common.py                 # YAML、路径、公共解析工具
├── schemas.py                # 公共 Schema 和组件 Schema
├── task_schemas.py           # mode/task 组合 Schema
├── normalize.py              # 结构规范化和默认值
├── validate.py               # 校验总入口
├── component_validators.py   # 组件语义校验
└── parser_refactor_plan.md   # 本方案
```

不要求一次性创建全部文件。推荐渐进迁移，避免同时破坏训练、测试和推理入口。

### 3.1 `common.py`

负责无任务差异的工具：

```text
load_yaml()
parse_args() 或公共命令行参数构造
relative_to_absolute_path()
路径字段处理工具
配置错误路径格式化工具
```

不负责：

```text
MLM/QA 判断
Loss/Metric 参数校验
训练/测试业务规则
```

### 3.2 `schemas.py`

保存固定结构模板，例如：

```text
COMMON_SCHEMA
MODEL_COMMON_SCHEMA
TOKENIZER_SCHEMA
DATALOADER_SCHEMA
LOSSES_SCHEMA
METRICS_SCHEMA
CUDA_SCHEMA
```

Schema 只描述：

```text
允许哪些字段
哪些字段必需
普通字段默认值
dict/list 的外层结构
```

不要在 Schema 中实现复杂的 `task` 条件判断。

### 3.3 `task_schemas.py`

保存模式和任务组合的配置结构：

```text
MLM_TRAIN_SCHEMA
QA_TRAIN_SCHEMA
MLM_TEST_SCHEMA
QA_TEST_SCHEMA
MLM_INFERENCE_SCHEMA
QA_INFERENCE_SCHEMA（需要时再添加）
```

每个 Schema 只包含该模式和任务真正使用的字段。

### 3.4 `normalize.py`

负责：

```text
选择 Schema
递归检查固定结构
补充普通字段默认值
保留多态 params 原样
规范化 mode/task
解析配置路径
```

不负责：

```text
ignore_index 的业务约束
ExactMatch 的 params 规则
QA 数据集 JSONL 内容校验
```

### 3.5 `validate.py`

负责组合校验：

```text
读取 mode 和 base_task
调用公共组件校验器
调用任务专属校验器
调用模式专属校验器
执行跨字段约束
```

### 3.6 `component_validators.py`

建议逐步拆成以下函数：

```text
validate_model()
validate_tokenizer()
validate_dataset()
validate_dataloader()
validate_losses()
validate_metrics()
validate_optimizer()
validate_scheduler()
validate_train()
validate_valid()
validate_test()
validate_checkpoint()
validate_cuda()
```

如果文件过长，再拆成：

```text
component_validators/
├── model.py
├── data.py
├── optimization.py
├── metrics.py
└── runtime.py
```

当前不需要一开始就拆成更多文件。

---

## 4. Schema 设计原则

### 4.1 训练、测试、推理分开

训练配置需要：

```text
model
tokenizer
dataset
dataloader
losses
metrics
optimizer
scheduler
train
valid
test
cuda
```

测试配置需要：

```text
name
task
checkpoint
dataset
dataloader
losses
metrics
cuda
```

推理配置根据现有 `inference_parser.py` 的真实 Builder 需求定义，不要直接复用训练 Schema。

测试配置不应被强制要求：

```text
optimizer
scheduler
train
valid
```

训练配置不应因为测试入口而强制要求：

```text
checkpoint
```

### 4.2 MLM 和 QA 的 Dataset Schema 分开

#### MLM Dataset

只包含 MLM 真正使用的字段：

```text
type
dataset_path
train_path
valid_path
test_path
max_seq_length
mlm_prob
```

#### QA Dataset

只包含 QA 当前 Builder 和 Dataset 真正使用的字段：

```text
type
dataset_path
train_path
valid_path
test_path
```

如果后续 QA Dataset 真正增加 `max_seq_length` 参数，再把它加入 QA Schema。不能因为 MLM 有 `max_seq_length`，就默认所有 QA 配置也必须有它。

#### QA Test Dataset

当前 QA Test Builder 只需要：

```text
type
dataset_path
test_path
```

若 QA 测试后续需要新的测试字段，再单独加入 QA Test Schema。

### 4.3 字段是否“存在”与字段是否“有效”分开

Schema/normalize 阶段负责：

```text
字段存在
字段是否允许出现
普通字段的默认值
基础 dict/list 结构
```

Validator 阶段负责：

```text
字符串是否非空
数字范围
支持项
路径存在性
字段间关系
任务语义
```

### 4.4 不使用 MLM 模板描述所有 Metric

Metrics 的公共结构是：

```text
metrics:
  train:
    - type
      params
  valid:
    - type
      params
  test:
    - type
      params
```

`params` 只在结构层被视为：

```text
必须存在的 dict 字段
```

`params` 内部不能使用 `TokenAccuracy` 模板递归补全。

具体规则放在 `validate_metrics()`：

```text
TokenAccuracy
→ ignore_index 规则

ExactMatch
→ 当前无参数规则

CharacterF1
→ 当前无参数规则
```

同理，Loss 的 `params` 也不应由普通递归深入处理。

---

## 5. 递归规范化器设计

现有 `check_and_add_config()` 可以被替换为一个只处理固定结构的函数，例如逻辑接口：

```text
normalize_node(config, schema, path)
```

其职责：

```text
dict 节点：检查字段集合，递归固定子节点
list 节点：检查列表类型和元素结构
普通叶子：缺失时填默认值，或检查必填
params 节点：只检查 dict，不递归内部
```

### 5.1 `params` 的处理

对于 Loss 和 Metric：

```text
params 字段不存在
→ 结构错误

params 为 None
→ 结构错误

params 不是 dict
→ 结构错误

params 是 dict
→ 保留原样，不进入固定模板递归
```

之后：

```text
validate_losses()
validate_metrics()
```

分别处理具体参数。

这样不再需要依赖：

```python
loss_completed
key.startswith("config.metrics")
```

其中 `key` 仍可保留作为错误路径，但不应再承担识别递归上下文的主要职责。

### 5.2 多态列表的处理

`losses` 和 `metrics` 的列表元素结构是稳定的，但元素内部 `params` 是多态的。

因此递归器只处理：

```text
列表存在
列表非空
元素为 dict
元素公共字段存在
```

并在 `params` 处停止深入。

---

## 6. 统一加载入口

最终底层加载接口建议为：

```text
load_config(yaml_path, mode)
```

处理流程：

```text
load_yaml()
→ 读取原始 task
→ 标准化 mode/base_task
→ 选择对应 Schema
→ normalize_config()
→ resolve_config_paths()
→ validate_config()
→ 返回字典配置
```

### 6.1 外部兼容接口

为了不立即修改所有调用方，可以保留：

```text
parse.parser.load_config(path)
→ load_config(path, mode="train")

parse.test_parser.load_config(path)
→ load_config(path, mode="test")

parse.inference_parser.load_config(path)
→ load_config(path, mode="inference")
```

这样 `start_train.py`、`start_test.py` 和推理入口无需同时重写。

### 6.2 测试 Builder 读取训练配置

QA Test Builder 当前会再次读取：

```text
checkpoint.config_path
```

并调用主 Parser 读取训练配置。

重构后应明确：

```text
测试配置
→ 使用 mode="test"

checkpoint 指向的训练配置
→ 使用 mode="train"
```

不能因为当前调用链方便，就用测试 Schema 解析训练配置。

---

## 7. 路径解析设计

路径解析应从配置结构校验中独立出来：

```text
normalize_config()
→ 保持相对路径或根据规则形成路径字段

resolve_config_paths()
→ 统一转为绝对 pathlib.Path

validate_paths()
→ 检查文件/目录存在性和类型
```

### 7.1 训练配置路径字段

根据具体 Schema 处理：

```text
model.model_path
tokenizer.vocab_file_path
dataset.dataset_path
dataset.train_path
dataset.valid_path
dataset.test_path
```

### 7.2 QA Test 路径字段

只处理当前真实使用的字段：

```text
checkpoint.path
checkpoint.config_path
dataset.dataset_path
dataset.test_path
```

不应让路径处理函数无条件访问：

```text
dataset.train_path
model.model_path
optimizer
```

### 7.3 相对路径基准

应明确记录：

```text
model.model_path、tokenizer.vocab_file_path、dataset.dataset_path
→ 相对于当前 YAML 所在目录

dataset.train_path、dataset.valid_path、dataset.test_path
→ 相对于 dataset.dataset_path
checkpoint.path、checkpoint.config_path
→ 相对于当前 YAML 所在目录
```

这个规则只能在路径解析层处理，不应散落到 Builder 中。

---

## 8. Validator 设计

### 8.1 公共 Validator

公共校验只检查组件自身语义：

```text
validate_model()
validate_tokenizer()
validate_dataloader()
validate_losses()
validate_metrics()
validate_cuda()
```

### 8.2 任务 Validator

任务 Validator 只检查任务相关约束：

```text
validate_mlm_config()
validate_qa_config()
```

例如：

```text
MLM
→ dataset.mlm_prob
→ MLM 数据集类型
→ MLM 模型类型

QA
→ QA 数据集类型
→ QA 模型类型
→ QA 训练/测试数据结构
```

不要把所有任务规则塞进公共 `validate_dataset()`。

### 8.3 模式 Validator

```text
validate_train_mode()
validate_test_mode()
validate_inference_mode()
```

例如：

```text
train
→ optimizer、scheduler、train 等字段

test
→ checkpoint、test 数据和 test metrics

inference
→ inference 专属字段
```

### 8.4 跨字段校验

集中在任务/模式校验的后面处理：

```text
model.hidden_size % model.num_heads == 0
```

只在双方字段都属于当前 Schema 时检查：

```text
dataset.max_seq_length <= model.max_position_embeddings
```

不能对不包含 `max_seq_length` 的 QA Schema 无条件访问该字段。

---

## 9. Metrics 和 Losses 的具体处理

### 9.1 Losses

普通结构层处理：

```text
losses 是非空 list
每个元素是 dict
每个元素包含 name/type/weight/params
params 是 dict
```

专属校验层处理：

```text
loss.type 是否受支持
loss.name 是否唯一
weight 是否有效
params 是否符合对应 loss.type
```

默认参数填充如果继续保留，应放在：

```text
normalize_losses()
```

而不是放在普通递归器内部。

### 9.2 Metrics

普通结构层处理：

```text
metrics.train/valid/test 的 list 结构
Metric 项的 type/params 字段
params 是 dict
```

专属校验层处理：

```text
Metric type 是否受支持
TokenAccuracy 的参数规则
ExactMatch 的参数规则
CharacterF1 的参数规则
```

如果某个 Metric 没有参数：

```text
params: {}
```

是合法配置。

### 9.3 不要求 Parser 静默忽略错误参数

当前是否显式拒绝未知参数，可以先保持已有项目决策，不在重构第一阶段扩大范围。

但最终应保证：

```text
Parser 成功
→ Builder 使用这些 params 不会产生明显的构造错误
```

---

## 10. 迁移路线

不建议一次性删除旧 Parser。推荐以下顺序。

### 阶段一：建立公共底层工具

新增：

```text
common.py
```

先迁移：

```text
load_yaml()
路径工具
```

保持现有 Parser 对外接口不变。

### 阶段二：抽取公共 Schema 和递归器

新增：

```text
schemas.py
normalize.py
```

先让新递归器支持：

```text
固定 dict/list
普通默认值
params 透明节点
错误路径
```

用现有 MLM 配置进行对比验证。

### 阶段三：拆分模式 Schema

建立：

```text
train schema
test schema
inference schema
```

确保测试配置不再要求训练字段，训练配置不再依赖测试字段。

### 阶段四：拆分 MLM/QA Dataset Schema

建立：

```text
MLM dataset schema
QA dataset schema
```

解决：

```text
QA 被迫携带 mlm_prob
QA 被迫携带 MLM 专属字段
MLM/QA 共用不兼容字段
```

### 阶段五：抽取组件 Validator

先抽取：

```text
validate_losses()
validate_metrics()
validate_dataset()
```

再抽取：

```text
validate_model()
validate_tokenizer()
validate_optimizer()
validate_scheduler()
```

### 阶段六：迁移 QA Test Parser

优先迁移 QA Test，因为它当前最能暴露：

```text
测试 Schema 与训练 Schema 混淆
QA Dataset 字段与 MLM Dataset 字段混淆
ExactMatch/CharacterF1 params 规则不一致
```

先确认：

```text
QA Test YAML 可以被解析
```

再运行 Builder。

### 阶段七：迁移 QA Train Parser

确认：

```text
QA Train YAML 不再要求 mlm_prob
QA Train 仍然能被 QA Builder 使用
```

### 阶段八：迁移 MLM Train/Test 和 Inference

逐个模式验证，确认 Builder 接口没有变化。

### 阶段九：删除旧逻辑

只有所有入口验证完成后，才删除：

```text
loss_completed
旧的重复 validate_config()
旧的 MLM 全局 Dataset Schema
parser.py/test_parser.py 中重复的组件校验
```

---

## 11. 兼容性要求

Parser 重构期间，Builder 的输出仍保持普通字典，不引入配置对象替代它们：

```text
config_dict["model"]
config_dict["dataset"]
config_dict["metrics"]
```

这样以下模块不需要因为 Parser 重构同步改写：

```text
main/tasks/MLM_PreTrain/builder.py
main/tasks/MLM_PreTrain/test_builder.py
main/tasks/QA_SFT/builder.py
main/tasks/QA_SFT/test_builder.py
main/start_test.py
main/Trainer/
main/Tester/
```

如果重构需要新增内部字段，例如：

```text
mode
base_task
```

应确保 Builder 不会因为顶层多出内部字段而受到影响；如果现有递归规则禁止未知字段，则应在返回前决定是否保留这些字段。

---

## 12. 测试策略

Parser 重构必须先测试配置解析，再测试 Builder，最后测试运行链路。

### 12.1 结构测试

验证：

```text
缺少必需字段
未知字段
错误 dict/list 类型
params 缺失
params 为 None
params 为非 dict
```

### 12.2 任务字段隔离测试

验证：

```text
MLM 配置缺少 mlm_prob → 按 MLM 规则处理
QA 配置不包含 mlm_prob → 可以通过
QA 配置不包含不使用的 max_seq_length → 可以通过
```

### 12.3 模式字段隔离测试

验证：

```text
QA Test 不要求 optimizer/scheduler/train/valid
MLM Train 不要求 checkpoint
```

### 12.4 多态 params 测试

验证：

```text
TokenAccuracy + ignore_index
ExactMatch + {}
CharacterF1 + {}
```

并确认：

```text
params 不会被错误套用 TokenAccuracy 模板
```

### 12.5 路径测试

验证：

```text
相对路径绝对化
dataset split 路径基于 dataset_path
checkpoint.config_path 默认推导
QA Test 只处理实际存在的路径字段
```

### 12.6 Builder 兼容测试

对每个模式和任务确认返回配置仍能被 Builder 使用：

```text
MLM Train
MLM Test
QA Train
QA Test
```

---

## 13. 完成标准

Parser 重构完成后应满足：

- `parser.py`、`test_parser.py`、`inference_parser.py` 不再复制整套校验逻辑；
- 配置模式、任务类型和组件结构明确分离；
- QA 不再被迫接受 MLM 专属字段；
- MLM 不会因为 QA 专属结构而失去原有配置能力；
- `losses.params` 和 `metrics.params` 不再被错误地套用固定模板递归；
- 普通递归器不再承载任务业务语义；
- 路径解析独立于语义校验；
- 训练、测试、推理使用统一底层加载流程；
- Builder 继续使用现有字典配置接口；
- 所有错误包含完整配置路径；
- 配置校验不修改传入配置；
- QA Test YAML 能在不携带 MLM 专属字段的情况下通过解析；
- MLM Train/Test 现有配置行为不被破坏。

---

## 14. 推荐的第一批实际改动

不要一开始重写所有校验。第一批建议只建立公共骨架：

```text
1. 抽取 common.py 中的 YAML 和路径工具
2. 定义 mode/base_task 标准化函数
3. 定义 train/test/inference 的最小 Schema 入口
4. 实现新的固定结构递归器
5. 让 params 成为不递归节点
6. 抽取 validate_losses() 和 validate_metrics()
7. 先迁移 QA Test 配置解析
```

第一批完成后，暂时保留旧的：

```text
parser.py
 test_parser.py
 inference_parser.py
```

通过新旧解析结果对比确认兼容性，再继续迁移 QA Train 和 MLM 配置。

---

## 15. 最终架构

最终配置处理链应为：

```text
入口 Parser
→ 确定 mode
→ 读取 task
→ 标准化 base_task
→ 选择 mode/task Schema
→ normalize 固定结构和默认值
→ resolve_config_paths 路径绝对化
→ validate 公共组件
→ validate 任务和模式
→ 执行跨字段约束
→ 返回字典 config
```

职责边界固定为：

```text
Schema
→ 配置形状

Normalize
→ 固定结构和普通默认值

Path Resolver
→ 路径绝对化

Component Validator
→ 组件自身语义

Task Validator
→ 任务专属语义

Mode Validator
→ 训练/测试/推理专属语义

Builder
→ 使用已经校验完成的配置构造运行组件
```

这个方案的核心不是增加更多配置类，而是把当前已经混在一起的职责拆开，并让 QA、MLM、训练、测试和推理分别拥有自己的有效配置范围。

---

# 十六、YAML 结构优化的详细方案

## 16.1 YAML 结构优化的目标

YAML 优化不是为了改变配置的书写风格，而是为了让 YAML 清楚表达三件事：

```text
当前运行模式是什么
当前业务任务是什么
当前组件真正需要哪些字段
```

最终目标是：

```text
QA 配置不携带 MLM 专属字段
MLM 配置不被 QA 专属字段污染
训练配置不要求测试专属字段
测试配置不要求训练专属字段
推理配置不要求训练或测试字段
Builder 读取路径保持稳定
```

## 16.2 保留当前的顶层组件式结构

当前建议继续使用平面组件式 YAML：

```yaml
name: ...
task: ...
model: ...
tokenizer: ...
dataset: ...
dataloader: ...
losses: ...
metrics: ...
optimizer: ...
scheduler: ...
train: ...
valid: ...
test: ...
checkpoint: ...
cuda: ...
```

不建议现在改成：

```yaml
mode:
  train:
    task:
      qa_sft:
```

也不建议把所有内容放入：

```yaml
tasks:
  qa_sft:
    ...
```

原因是当前 Builder 已经直接读取：

```text
config_dict["model"]
config_dict["dataset"]
config_dict["metrics"]
config_dict["losses"]
```

保留顶层组件式结构，可以让 Parser 重构只改变“哪些字段合法”，而不强迫 Builder 同时改成新的嵌套访问方式。

## 16.3 运行模式与任务名称

当前外部任务名称已经被入口和 Registry 使用：

```text
mlm_pretrain
mlm_pretrain_test
qa_sft
qa_sft_test
```

第一阶段不必修改 YAML 中的 `task` 写法，也不必马上新增顶层 `mode` 字段。

Parser 内部可以把任务标准化为：

```text
原始 task
→ mode
→ base_task
```

映射关系：

```text
mlm_pretrain
→ mode=train, base_task=mlm_pretrain

mlm_pretrain_test
→ mode=test, base_task=mlm_pretrain

qa_sft
→ mode=train, base_task=qa_sft

qa_sft_test
→ mode=test, base_task=qa_sft
```

这样校验逻辑只需要判断：

```text
mode == "train" / "test" / "inference"
base_task == "mlm_pretrain" / "qa_sft"
```

而不是在每个校验分支里重复判断四个完整任务名称。

如果未来希望让 YAML 更明确，再考虑增加：

```yaml
mode: test
task: QA_SFT
```

但这属于第二阶段改动，不能在第一阶段和 Parser 内部重构同时扩大范围。

## 16.4 四类核心 YAML

当前应明确设计四类核心配置：

```text
MLM Train
QA Train
MLM Test
QA Test
```

Inference 继续根据当前 `inference_parser.py` 和对应 Builder 的真实需求单独定义，不应直接复制 Train 或 Test Schema。

### 16.4.1 MLM Train

MLM Train 可以包含：

```text
name
task
model
tokenizer
dataset
dataloader
losses
metrics
optimizer
scheduler
train
valid
test
cuda
```

MLM Dataset 中真正属于 MLM 的字段：

```text
type
dataset_path
train_path
valid_path
test_path
max_seq_length
mlm_prob
```

`mlm_prob` 只能由 MLM 的 Schema 和校验器负责。它对应：

```text
MLMCollator(mlm_prob=...)
```

### 16.4.2 QA Train

QA Train 可以包含：

```text
name
task
model
tokenizer
dataset
dataloader
losses
metrics
optimizer
scheduler
train
valid
test
cuda
```

当前 QA Dataset 真正使用：

```text
type
dataset_path
train_path
valid_path
test_path
```

当前 `QADataset` 和 `QA_SFT.builder.py` 不使用：

```text
mlm_prob
```

当前 `QADataset` 构造函数也不接收：

```text
max_seq_length
```

因此在 QA 当前 Schema 中不应要求这两个字段。以后 QA Dataset 真正增加长度参数时，再根据实际 Builder 接口加入 QA Schema。

### 16.4.3 MLM Test

MLM Test 当前需要：

```text
name
task
checkpoint
dataset
dataloader
losses
metrics
cuda
```

MLM Test Dataset 需要：

```text
type
dataset_path
test_path
max_seq_length
mlm_prob
```

因为当前 `MLM_PreTrain.test_builder` 会使用测试配置中的 Dataset 和 Loss，同时从训练配置中读取 Tokenizer 和 Model。

### 16.4.4 QA Test

QA Test 当前需要：

```text
name
task
checkpoint
dataset
dataloader
losses
metrics
cuda
```

QA Test Dataset 当前只需要：

```text
type
dataset_path
test_path
```

QA Test 不应要求：

```text
train_path
valid_path
max_seq_length
mlm_prob
optimizer
scheduler
train
valid
```

当前 QA Test Builder 已经使用：

```text
config_dict["dataset"]["test_path"]
```

而不是 `train_path`。

## 16.5 `model` 字段的处理

当前 MLM 和 QA 都使用 Transformer 模型配置，但最终应由任务 Schema 决定模型字段集合，而不是仅凭模型目录模板决定。

公共模型字段可以包括：

```text
type
name
model_path
hidden_size
max_position_embeddings
type_vocab_size
num_heads
intermediate_size
num_layers
embedding_dropout
attention_dropout
hidden_dropout
layer_norm_eps
backbone_path
```

但是否要求每个字段，应以对应模型 Builder 的构造函数为准。

Parser 不应因为某个 MLM 模型需要字段，就无条件要求所有 QA 模型也必须使用同样字段。更长期的方案是：

```text
model.type → 选择模型族
model.name → 选择具体模型
model.name 对应模型 Schema/Validator
```

第一阶段可以继续使用任务 Schema，第二阶段再按模型名称细分。

## 16.6 `tokenizer` 字段的处理

当前 QA 和 MLM 都通过同一类 Tokenizer Registry 构造 Tokenizer，因此可以先保留公共 Tokenizer Schema：

```text
type
vocab_file_path
do_lower_case
strip_accents
max_input_chars_per_word
```

但是 Parser 仍要区分：

```text
Tokenizer 配置结构合法
Tokenizer 是否受当前任务支持
```

Tokenizer 文件路径必须由路径解析层转换为绝对 `Path`，语义校验由 Tokenizer Validator 处理。

## 16.7 `dataset` 字段的处理

Dataset 是当前 YAML 重构最需要拆分的组件。

### 公共 Dataset 字段

可以先保留：

```text
type
dataset_path
```

### Split 字段

按模式和任务决定：

```text
MLM Train:
    train_path, valid_path, test_path

QA Train:
    train_path, valid_path, test_path

MLM Test:
    test_path

QA Test:
    test_path
```

### 任务专属字段

```text
MLM:
    max_seq_length
    mlm_prob

QA:
    当前没有上述两个字段
```

### 路径解析

路径解析必须根据 Schema 中实际存在的字段进行，不能无条件访问：

```text
dataset.train_path
dataset.valid_path
dataset.test_path
```

QA Test 当前只解析：

```text
dataset.dataset_path
dataset.test_path
```

Dataset 文件相对路径规则保持：

```text
dataset.dataset_path
→ 相对于当前 YAML 文件目录

dataset.train_path、valid_path、test_path
→ 相对于 dataset.dataset_path
```

## 16.8 `dataloader` 字段的处理

公共 DataLoader 字段可保持：

```text
batch_size
num_workers
pin_memory
```

训练 DataLoader 额外允许：

```text
shuffle
drop_last
```

测试 DataLoader 当前不需要强制要求：

```text
shuffle
drop_last
```

因为当前测试 Builder 未显式读取它们，PyTorch 默认行为已经是：

```text
shuffle=False
drop_last=False
```

如果未来希望 YAML 明确表达测试行为，可以加入测试 Schema，但不能让训练专属字段自动成为测试必填字段。

## 16.9 `losses` 字段的处理

Loss 外层结构保持：

```yaml
losses:
  - name: ...
    type: ...
    weight: ...
    params: ...
```

通用递归只处理：

```text
列表结构
公共字段存在
params 是 dict
```

不进入 `params` 内部递归。

`validate_losses()` 根据 `loss.type` 处理：

```text
CrossEntropyLoss
→ ignore_index
```

Loss 默认值策略必须明确：

```text
整个 losses 缺失
→ 根据当前模式/任务决定是否允许

loss 项存在但 params 为空
→ 不使用不匹配的其他 Loss 参数模板
→ 由 validate_losses() 判断
```

## 16.10 `metrics` 字段的处理

Metrics 外层结构可以在 Train/Test 之间复用：

```yaml
metrics:
  train:
    - type: ...
      params: ...
  valid:
    - type: ...
      params: ...
  test:
    - type: ...
      params: ...
```

但具体要求由模式决定：

```text
Train 通常需要 train/valid/test
Test 只需要 test
Inference 根据实际推理需求决定
```

`params` 统一作为透明字典：

```text
params 字段存在
params 是 dict
params 不进入普通固定模板递归
```

具体参数由 `validate_metrics()` 根据 Metric 类型处理：

```text
TokenAccuracy
→ ignore_index

ExactMatch
→ 当前允许空 dict

CharacterF1
→ 当前允许空 dict
```

不要把 `TokenAccuracy.params` 作为所有 Metric 的默认模板。

## 16.11 `checkpoint` 字段的处理

Checkpoint 是 Test 的核心配置，不应出现在普通 Train Schema 中作为必需字段：

```yaml
checkpoint:
  path: ...
  config_path: ...
```

其中：

```text
path
→ 当前测试使用的 checkpoint

config_path
→ 对应训练配置
→ 可以允许 null，然后按现有规则推导
```

QA Test Builder 当前会读取：

```text
config_dict["checkpoint"]["config_path"]
```

并用训练 Parser 加载训练配置。因此重构后必须保持：

```text
测试配置使用 test Schema
训练 checkpoint 配置使用 train Schema
```

## 16.12 `optimizer`、`scheduler`、`train`、`valid`、`test`

这些组件不能在所有模式中使用同一组必需规则。

```text
optimizer/scheduler/train
→ 训练模式

valid
→ 训练模式，根据 do_validation 决定运行与否

test
→ 训练配置中表示是否在训练流程后测试
→ 独立测试配置中不应被当成训练控制块
```

Test Parser 不应因为复用训练模板而要求：

```text
optimizer
scheduler
train
valid
```

## 16.13 YAML 迁移检查表

每个 YAML 迁移到新 Schema 前，按以下顺序检查：

```text
1. 确定 mode
2. 确定 base_task
3. 确定当前 Builder
4. 列出 Builder 实际读取的配置路径
5. 保留这些字段
6. 删除其他任务专属字段
7. 检查 params 是否为 dict
8. 检查路径相对基准
9. 运行 Parser 结构校验
10. 再运行 Builder
```

不要根据注释判断字段是否必需，必须以当前 Builder、Dataset、Task 和入口的实际读取代码为准。

---

# 十七、Parser 重构的详细实现流程

## 17.1 第一阶段：备份旧 Parser

在重构前由开发者自行备份：

```text
parser.py
test_parser.py
inference_parser.py
```

备份文件只用于对照，不参与运行，不应被自动导入。

## 17.2 第二阶段：抽取公共工具

先抽取：

```text
load_yaml()
relative_to_absolute_path()
relative_to_absolute_path_full_dict()
```

确保三个入口使用同一套基础工具。

## 17.3 第三阶段：定义模式和任务标准化

实现逻辑：

```text
raw task
→ mode
→ base_task
```

如果任务名称不受支持，在标准化阶段就抛出包含完整字段路径的错误。

## 17.4 第四阶段：定义四套核心 Schema

先定义：

```text
MLM_TRAIN_SCHEMA
QA_TRAIN_SCHEMA
MLM_TEST_SCHEMA
QA_TEST_SCHEMA
```

不要先追求支持所有未来任务，也不要先建立复杂 Schema DSL。

## 17.5 第五阶段：实现固定结构 normalize

新的 normalize 只负责：

```text
dict/list 类型
允许字段
必需字段
普通默认值
params 透明节点
错误路径
```

它不负责：

```text
任务语义
Loss 参数值
Metric 参数值
文件内容
模型构造
```

## 17.6 第六阶段：实现路径解析

根据选中的 Schema 只处理当前存在的路径字段：

```text
相对路径
→ 绝对 pathlib.Path
```

路径解析之后再进行路径存在性校验。

## 17.7 第七阶段：抽取组件 Validator

先实现：

```text
validate_dataset()
validate_losses()
validate_metrics()
```

再实现：

```text
validate_model()
validate_tokenizer()
validate_dataloader()
validate_checkpoint()
validate_optimizer()
validate_scheduler()
```

## 17.8 第八阶段：迁移 QA Test

QA Test 是第一条迁移链路，因为它最容易暴露：

```text
测试字段与训练字段混淆
QA Dataset 与 MLM Dataset 混淆
ExactMatch/CharacterF1 的 params 规则问题
```

先验证：

```text
QA Test YAML
→ test Schema
→ normalize
→ resolve paths
→ validate
```

再验证：

```text
QA Test Builder
```

## 17.9 第九阶段：迁移 QA Train

确认 QA Train YAML 不再被要求填写：

```text
mlm_prob
```

也不被无条件要求填写当前 QA Builder 没有使用的字段。

## 17.10 第十阶段：迁移 MLM Train/Test

保留 MLM 当前真实使用的：

```text
max_seq_length
mlm_prob
```

确保迁移后 `MLMCollator` 仍能获得 `mlm_prob`。

## 17.11 第十一阶段：迁移 Inference

Inference 单独读取当前入口和 Builder 的真实配置需求，不直接复制 Test Schema。

## 17.12 第十二阶段：清理旧逻辑

只有四类配置都能通过解析和 Builder 构造后，才删除：

```text
loss_completed
旧 CUSTOM_DICT 的任务混合字段
重复 validate_config()
重复路径处理
parser.py/test_parser.py/inference_parser.py 中的重复逻辑
```

---

# 十八、YAML 与 Builder 的兼容约束

Parser 重构不能改变 Builder 预期的字典访问方式，除非同步修改并验证 Builder。

当前必须保持：

```text
config_dict["model"]
config_dict["tokenizer"]
config_dict["dataset"]
config_dict["dataloader"]
config_dict["losses"]
config_dict["metrics"]
```

QA Test Builder 当前还依赖：

```text
config_dict["checkpoint"]["config_path"]
config_dict["dataset"]["test_path"]
config_dict["dataloader"]
config_dict["losses"]
config_dict["metrics"]["test"]
```

Parser 可以增加内部元数据，但不应让 Builder 因配置层级改变而被迫重写。

## 18.1 配置字段消费者登记

重构前建议为每个字段登记真实消费者：

```text
字段
→ 哪个 Parser Schema 允许
→ 哪个 Builder 读取
→ 哪个组件使用
→ 是否有默认值
→ 是否需要路径解析
→ 是否需要语义校验
```

例如：

```text
mlm_prob
→ MLM Train/Test Schema
→ MLM Test/Train Builder
→ MLMCollator
→ 默认 0.15
→ 否
→ 校验范围 (0,1)
```

```text
max_seq_length
→ 当前 MLM Schema
→ MLM Dataset/Builder
→ MLM Dataset
→ 默认 128
→ 否
→ 校验正整数
```

```text
answers_list
→ 不属于 YAML
→ QA Dataset/Collator 运行时生成
→ QA Metrics Process 使用
→ 不参与 Parser
```

## 18.2 不把运行时字段写进 YAML

以下字段是运行时 Batch 或 Outputs 字段，不属于 YAML：

```text
predictions
references
answers_list
start_logits
end_logits
offset_mapping_list
```

YAML 只配置：

```text
如何构造任务和指标
```

不配置：

```text
某个具体 Batch 中运行时才产生的中间结果
```

---

# 十九、Parser 重构的验证矩阵

每一个模式和任务组合都需要分别验证。

## 19.1 解析层验证

```text
MLM Train YAML → train/mlm_pretrain
QA Train YAML → train/qa_sft
MLM Test YAML → test/mlm_pretrain
QA Test YAML → test/qa_sft
```

确认：

```text
允许字段正确
必需字段正确
默认值只填入当前 Schema 字段
路径字段正确绝对化
params 不被错误递归
```

## 19.2 负例验证

```text
QA YAML 缺少 mlm_prob → 不因该字段报错
QA YAML 写入不允许的 MLM 字段 → 按 Schema 决定是否报错
QA Test 缺少 train_path → 可以通过结构层
QA Test 缺少 checkpoint.path → 报错
MLM 配置缺少 mlm_prob → 按 MLM Schema 报错或使用明确默认值
Test 配置出现 optimizer → 按 Test Schema 报错
```

“QA 可以不填写”与“QA 可以随便填写”是两件事：

```text
不要求无关字段
≠
允许任意未知字段
```

Schema 仍应阻止拼写错误和明显无关字段。

## 19.3 Builder 验证

解析成功后，逐一调用：

```text
MLM Train Builder
MLM Test Builder
QA Train Builder
QA Test Builder
```

只确认组件能够构造，不立即执行完整训练或测试。

## 19.4 运行链路验证

Builder 构造验证通过后，再分别进行：

```text
最小 Batch forward
Metric update
Metric compute
Checkpoint load
```

最后才运行完整训练、验证、测试或推理。

---

# 二十、当前推荐的最终 YAML 设计结论

```text
保留顶层组件式 YAML
不立即增加 mode 字段
不改成 task 深层嵌套
用 mode + base_task 选择 Schema
MLM/QA Dataset Schema 分开
Train/Test Schema 分开
Inference Schema 单独维护
params 只做透明 dict
Loss/Metric 语义由 Validator 负责
路径解析根据 Schema 中真实存在的字段执行
Builder 继续接收平面字典
```

YAML 重构的本质是：

```text
让配置只描述当前任务真正使用的组件
让 Parser 只接受当前模式真正允许的字段
让 Builder 继续读取稳定的组件路径
```
