# 通用 Parser 重构实施方案

## 1. 文档定位

本文是后续 Parser 实现的主设计文档。实现时以本文定义的接口、数据流、职责边界和迁移顺序为准。

这次重构允许同步修改：

```text
YAML
Parser
Registry
Builder
Task
入口
组件构造接口
配置快照
Test/Inference 配置加载方式
```

不再以兼容旧扁平配置为主要目标。旧 `parse.copy` 和旧 YAML 只作为迁移对照，不决定新架构。

最终目标不是让 Parser 特别支持 MLM、QA、GPT，而是：

> Parser 支持任意遵守 ModeSpec、TaskSpec、ComponentSpec 和 SchemaNode 协议的扩展。

MLM、QA 只是第一批内置扩展。未来接入 GPT 时，不应修改 Parser 引擎。

---

## 2. 当前问题

当前真实运行入口使用：

```text
start_train.py     → parse.copy.parser
start_test.py      → parse.copy.test_parser
start_inference.py → parse.copy.inference_parser
```

当前问题：

```text
三套 Parser 重复 YAML、路径和校验逻辑
mode 被编码进 task 名称
全局 CUSTOM_DICT 淋合任务和模式
Parser 硬编码组件参数
params 多态结构被固定模板处理
路径字段通过字段名硬编码
Builder 与旧扁平路径强耦合
Registry 只登记运行对象，没有统一配置规格
新增任务需要修改 Parser 条件分支
```

新系统必须消除这些问题，而不是只拆分旧函数。

---

## 3. 理想 YAML

### 3.1 顶层结构

```yaml
version: 1

experiment:
  name: TinyBert_QA_SFT
  seed: 42

run:
  mode: train
  resume_from: null

task:
  type: qa_sft
  params: null

components:
  tokenizer:
    type: qa_wordpiece
    params: null

  model:
    type: tinybert_qa
    params: null
    init_from:
      kind: backbone
      path: ../experiments/mlm/checkpoints/best.pt

  losses:
    - name: qa_cross_entropy
      type: cross_entropy
      weight: 1.0
      params: null

data:
  dataset:
    type: cmrc2018
    root: ../dataset/QA/SFT/CMRC2018/tiny_cmrc2018
    splits:
      train: tiny_cmrc2018_train.jsonl
      valid: tiny_cmrc2018_val.jsonl
      test: tiny_cmrc2018_test.jsonl
    params: null

  collator:
    type: qa_span
    params: null

  loaders:
    train:
      batch_size: 8
      shuffle: true
      num_workers: 0
      pin_memory: false
      drop_last: false
    valid:
      batch_size: 8
      shuffle: false
      num_workers: 0
      pin_memory: false
      drop_last: false
    test:
      batch_size: 8
      shuffle: false
      num_workers: 0
      pin_memory: false
      drop_last: false

optimization:
  optimizer:
    type: adamw
    params: null

  scheduler:
    type: step_lr
    interval: step
    params: null

  loop:
    total_steps: 50000
    accumulation_steps: 1
    log_every_steps: 100

  validation:
    enabled: true
    every_steps: 500
    early_stopping:
      enabled: true
      patience: 10
    checkpoint:
      save_every_rounds: 1

evaluation:
  test_after_train: true
  metrics:
    train:
      - type: exact_match
        params: null
      - type: character_f1
        params: null
    valid:
      - type: exact_match
        params: null
      - type: character_f1
        params: null
    test:
      - type: exact_match
        params: null
      - type: character_f1
        params: null

runtime:
  device:
    type: cuda
    index: null
    count: 1
```

### 3.2 顶层职责

```text
version       → 配置协议版本
experiment    → 实验身份与随机种子
run           → 本次运行模式和状态来源
task          → 业务任务及任务参数
components    → 可构造算法组件
data          → 数据集、Collator 和 Split Loader
optimization  → 训练模式专属优化与循环
evaluation    → Metrics 和训练后测试
inference     → 推理模式专属参数
runtime       → 设备及运行环境
```

### 3.3 Mode 与 Task 分离

```yaml
run:
  mode: test

task:
  type: qa_sft
```

不再使用：

```text
qa_sft_test
mlm_pretrain_test
mlm_pretrain_inference
```

系统查找键统一为：

```text
(mode_key, task_key)
```

---

## 4. 总体架构

完整链路：

```text
YAML
→ load_raw_yaml
→ bootstrap_config
→ collect_specs
→ compile_schema
→ normalize_structure
→ normalize_polymorphic_nodes
→ apply_overrides
→ resolve_paths
→ validate_config
→ ParsedConfig
→ BuildContext
→ Mode Runner + Task Builder
```

这里每个箭头都表示“上一步返回对象，作为下一步输入”，不是只描述概念：

```text
load_raw_yaml
输入：YAML 文件路径
返回：raw_config: dict
原因：先安全地把文本变成 Python 对象，不做任务判断

bootstrap_config
输入：raw_config、config_path、Registries、expected_mode
返回：ParseContext
原因：先取得 mode、task 和所有 component.type，才能知道后面要组合哪些 Schema

collect_specs
输入：ParseContext、ParserServices
返回：SpecBundle
原因：把本次真正使用的 Root/Mode/Task/Component Spec 集中起来

compile_schema
输入：SpecBundle
返回：CompiledSchema
原因：把多个 Schema contribution 合成一份最终契约，并提前收集路径、多态和 Validator 元数据

normalize_structure
输入：raw_config、CompiledSchema、ParseContext
返回：新的 normalized_config: dict
原因：统一缺失字段、默认值、None、dict/list 结构和 canonical key

normalize_polymorphic_nodes
输入：normalized_config、CompiledSchema、ParseContext
返回：新的 config: dict
原因：按 component.type 处理 params，保证最终 params 都是 dict

apply_overrides
输入：config、ConfigOverride 列表、CompiledSchema
返回：新的 config: dict
原因：CLI 覆盖也必须经过同一 Schema，不得绕过配置规则

resolve_paths
输入：config、CompiledSchema、ParseContext
返回：路径已绝对化的新 config: dict
原因：统一处理 config_dir、字段基准和推导路径

validate_config
输入：config、CompiledSchema、ParseContext
返回：None
原因：只检查业务语义和运行环境；失败直接抛异常

ParsedConfig
输入：最终 config、context、compiled_schema、fingerprint
返回：一个数据对象
原因：同时交付 Builder 需要的配置和调试/复现需要的上下文

BuildContext
输入：ParsedConfig、logger、实验目录、设备信息、ParserServices
返回：Task Builder 使用的运行上下文
原因：运行依赖不应该塞进 YAML 或 Parser

Mode Runner + Task Builder
输入：BuildContext
返回：TrainSummary/TestSummary/InferenceResult 等运行结果
原因：Runner 管模式流程，Builder 管任务组件组装
```

所有返回配置的新阶段都不得修改输入对象；`validate_config()` 成功返回 `None`，失败抛 `ConfigError` 子类。

Parser 引擎不包含：

```text
if task == mlm
if task == qa
if model == tinybert
if optimizer == adamw
```

这些差异全部来自 Registry 中的 Spec。

---

## 5. 核心数据类型

本节中的对象都是“数据载体”，不是完整流程。每个对象的创建者和消费者如下。

## 5.1 `ConfigPath`

本质：表示 YAML 配置树中一个节点的逻辑位置，不是磁盘文件路径。

内部数据：

```text
parts: tuple[str | int, ...]
```

例如：

```text
("config", "components", "losses", 0, "params")
→ 显示为 config.components.losses[0].params
```

创建和变换：

```text
ConfigPath.root("config")
→ 返回 ConfigPath(("config",))

path.child("components")
→ 返回新的 ConfigPath
→ 原 path 不变

path.index(0)
→ 返回新的 ConfigPath，并加入列表索引

str(path)
→ 返回用户可读字符串
```

创建者：Schema 递归、Bootstrap 扫描、Override 解析。

消费者：错误对象、`ParseContext.selections`、Schema walker、路径依赖图、Validator。

`ConfigPath` 不提供 `exists()`；某路径是否在配置或映射中存在，由操作该容器的函数判断。

## 5.2 `ParseContext`

本质：Bootstrap 完成后生成的只读“本次解析选择结果”。它不是最终配置。

字段：

```text
config_path: pathlib.Path
config_dir: pathlib.Path
expected_mode: str | None
mode_key: str
task_key: str
selections: Mapping[ConfigPath, ComponentSelection]
validate_runtime: bool
```

创建者：

```text
bootstrap_config() → ParseContext
```

消费者：

```text
collect_specs
compile_schema
normalize_polymorphic_nodes
resolve_paths
validate_config
Runner/Builder
```

`selections` 记录：

```text
config.components.model → tinybert_qa
config.components.losses[0] → cross_entropy
config.evaluation.metrics.valid[0] → exact_match
```

方法：

```text
get_selection(path)
→ 实际执行 self.selections[path]
→ 找不到时抛 BootstrapError
```

它查询的是 selections 字典中是否有该键，不是调用 `path.exists()`。

## 5.3 `ComponentSelection`

本质：表示“Bootstrap 在某个 YAML 节点选择了哪个 ComponentSpec”。

字段：

```text
category
key
path
spec
```

示例：

```text
category="model"
key="tinybert_qa"
path=ConfigPath(("config", "components", "model"))
spec=<ComponentSpec tinybert_qa>
```

创建者：Bootstrap 在读取并解析 `type` 后创建。

消费者：

```text
collect_specs
normalize_polymorphic_nodes
Component Validator
Builder
```

保存 `spec` 的原因：类型已经在 Bootstrap 中确认，后续阶段不应反复查 Registry 或再次解释别名。

## 5.4 `ConfigOverride`

本质：表示“一次尚未应用的配置覆盖请求”。

字段：

```text
path: ConfigPath
raw_value: Any
source: str
```

例如：

```text
path=config.run.resume_from
raw_value="../checkpoints/last.pt"
source="cli:--resume"
```

创建者：CLI 参数解析器或超参数运行器。

消费者：`apply_overrides()`。

它自身不修改配置。`apply_overrides()` 找到目标 SchemaNode，使用该节点的 `normalize()` 处理 `raw_value` 后，才把结果写入新的配置副本。

---

## 6. SchemaNode 协议

Schema 是一棵由节点对象组成的树。YAML 解析后也是一棵由 dict、list 和标量组成的树。`normalize()` 的工作就是让 Schema 树与 YAML 值树从根节点开始一一对应处理。

## 6.1 `normalize()` 到底如何被调用

外部只调用一次根节点：

```text
compiled_schema.root.normalize(
    raw_config,
    ConfigPath.root("config"),
    context,
)
```

后续递归由容器节点负责：

```text
MappingSchema
→ 对每个字段调用对应 child_schema.normalize(...)

SequenceSchema
→ 对每个列表元素调用 item_schema.normalize(...)

ScalarSchema
→ 不再递归，直接处理当前值
```

因此 `SchemaNode` 抽象基类只声明：

```text
normalize(self, value, path, context) → Any
```

`self` 是当前规则，`value` 是当前 YAML 值，`path` 是当前逻辑位置，`context` 是 Bootstrap 上下文。

统一返回规则和异常类型：

```text
成功 → 返回规范化值
可选字段缺失且无默认 → 返回 MISSING
配置值的结构/类型/缺失/空值/注册键错误
→ 按具体情形抛 MissingFieldError、NullNotAllowedError、TypeMismatchError、UnknownFieldError、UnknownRegistryKeyError 或其他 ConfigError 子类
Schema 节点定义参数非法
→ 在构造阶段抛 TypeError 或 ValueError，不进入 normalize()
```

统一副作用规则：不修改传入值。

`merge` 和 `walk` 不作为节点实例方法：

```text
merge_schema() → schema/merge.py 的纯函数
walk_schema()  → schema/walk.py 的纯函数
```

## 6.2 `ScalarSchema.normalize()`

字段：

```text
python_types
required
default=MISSING
allow_none
normalizer=None
```

调用者：父 `MappingSchema`、父 `SequenceSchema`，或者根节点测试。

执行顺序、返回值和异常类型：

```text
value is MISSING
├── default != MISSING → 返回 deepcopy(default)
├── required=True      → 抛 MissingFieldError
└── 否则               → 返回 MISSING

value is None
├── allow_none=True → 返回 None
└── 否则            → 抛 NullNotAllowedError

value 是具体值
├── 类型不匹配 → 抛 TypeMismatchError
├── 有 normalizer → 调用 normalizer(value)
│   ├── 正常返回 → 返回转换结果
│   ├── 抛 ConfigError 子类 → 原样向上传递
│   ├── 抛 ValueError/TypeError → 包装为 ParameterNormalizationError
│   └── 其他异常 → 原样向上传递，视为 normalizer 实现缺陷
└── 否则 → 返回原 value
```

节点构造参数本身非法时，不进入 `normalize()`：参数类型错误抛 `TypeError`，参数取值或组合错误抛 `ValueError`。

`normalizer` 只做明确且无歧义的转换，例如字符串 `strip()`；不能把 `"one"` 猜成数字 1。

## 6.3 `MappingSchema.normalize()`

字段：

```text
fields: Mapping[str, SchemaNode]
allow_unknown: bool = false
```

必填信息属于各子节点的 `required`，不再额外维护容易冲突的 `required: set[str]`。

输入：当前 YAML dict；返回：新 dict。

执行顺序：

```text
1. value 不是 Mapping → TypeMismatchError
2. unknown = value.keys - fields.keys
3. 不允许未知字段且 unknown 非空 → UnknownFieldError
4. result = {}
5. 遍历 fields
6. raw_child = value.get(field_name, MISSING)
7. normalized_child = child_schema.normalize(
       raw_child,
       path.child(field_name),
       context,
   )
8. normalized_child is not MISSING 时写入 result
9. 若允许未知字段，deepcopy 后写入 result
10. 返回 result
```

完整示例：

```text
Schema fields:
name → required str
seed → optional int, default=42

YAML value:
{"name": "demo"}

返回：
{"name": "demo", "seed": 42}
```

## 6.4 `SequenceSchema.normalize()`

字段：

```text
item_schema: SchemaNode
required: bool = false
default: Any = MISSING
min_items: int = 0
max_items: int | None = None
allow_none: bool = false
```

`required`、`default`、`allow_none` 的语义与 `ScalarSchema` 一致：

```text
required=True 且 default is not MISSING
→ Schema 定义错误，构造阶段抛 ValueError

default is not MISSING 且 default 不是 list
→ Schema 定义错误，构造阶段抛 TypeError

default is None 且 allow_none=False
→ Schema 定义错误，构造阶段抛 ValueError

default 是 list 但长度不满足 min_items/max_items
→ Schema 定义错误，构造阶段抛 ValueError

default 是 list 但存在不符合 item_schema 的元素
→ Schema 定义错误，构造阶段抛 ValueError 或 SchemaError
→ 建议在 Schema 构造阶段执行无用户路径的默认值结构检查

字段不存在且 default is not MISSING
→ 返回 default 的 deepcopy

字段不存在且 required=True 且无 default
→ 抛 MissingFieldError

字段不存在且 required=False 且无 default
→ 返回 MISSING

字段存在且 value is None 且 allow_none=True
→ 返回 None

字段存在且 value is None 且 allow_none=False
→ 抛 NullNotAllowedError
```

输入：当前 YAML list；返回：新 list。

执行顺序：

```text
1. value is MISSING
   ├── default is not MISSING → 返回 deepcopy(default)
   ├── required=True → MissingFieldError
   └── required=False → 返回 MISSING
2. value is None
   ├── allow_none=True → 返回 None
   └── allow_none=False → NullNotAllowedError
3. value 不是 list → TypeMismatchError
4. len(value) < min_items 或 max_items is not None 且 len(value) > max_items
   → SchemaError
5. result = []
6. enumerate(value)
7. item_schema.normalize(item, path.index(i), context)
8. 子结果不是 MISSING 时 append 到 result
9. 返回 result
```

`default` 是 Schema 定义者提供的、已经符合该 `SequenceSchema` 输出契约的默认列表，不在 `normalize()` 中再次调用 `item_schema.normalize()`。构造阶段必须检查：

```text
default is MISSING
→ 没有默认值

default 不是 list
→ TypeError

default 是 list 且长度小于 min_items
→ ValueError

default 是 list 且 max_items is not None 且长度超过 max_items
→ ValueError
```

`default=None` 不属于 `SequenceSchema` 的合法默认值，即使 `allow_none=True` 也不例外。`allow_none` 只约束用户或 Override 实际传入的 `value`；它不改变 `default` 的类型契约。

缺失字段使用默认值时，`normalize()` 只返回 `deepcopy(default)`，不重新检查列表长度，也不调用 `item_schema.normalize()`。因此 Schema 定义者必须保证默认列表及其元素符合 `item_schema`；默认值不经过 `normalizer` 或子 Schema 处理。

列表元素的 `MISSING` 结果不进入返回列表，`None` 结果则保留。用户实际提供的列表会逐元素调用 `item_schema.normalize()`，并保留带索引的错误路径。

示例：Loss 列表第 0 项错误时，子节点收到路径：

```text
config.components.losses[0]
```

## 6.5 `EnumSchema.normalize()`

本质：有限字符串的 ScalarSchema。

输入：例如 `"STEP"`；返回：canonical 字符串 `"step"`。

执行：

```text
基础缺失/None/类型处理
→ 根据 case_sensitive 规范化查询键
→ 在 choices 中查询
→ 找到 → 返回 choices 映射的 canonical value
→ 未找到 → 抛 SchemaError；消息包含实际值、允许值和 path
```

适合：

```text
scheduler.interval
model.init_from.kind
```

`run.mode` 由 Mode Registry 解析，不维护重复 Enum。

## 6.6 `RegistryKeySchema.normalize()`

本质：把 YAML 中组件类型名称变成 Registry canonical key。

输入：例如 `"AdamW"`；返回：`"adamw"`。

执行、返回值和异常类型：

```text
value is MISSING 且 required=True 且无 default
→ MissingFieldError

value is None 且 allow_none=False
→ NullNotAllowedError

value 不是 str
→ TypeMismatchError

value 是空字符串或去除首尾空白后为空
→ SchemaError；消息包含 path 和“键不能为空”

value 是非空字符串
→ registry.resolve_key(value)
├── 成功 → 返回 canonical key
└── 未注册 → 保留并向上传递 UnknownRegistryKeyError
```

Registry 负责别名，不由 Schema 自己维护 `aliases_allowed`。

## 6.7 `PolymorphicSchema.normalize()`

用于：

```text
{type, params}
```

字段：

```text
category
discriminator="type"
params_field="params"
common_schema
```

第一阶段 `normalize()`：

```text
调用 common_schema.normalize(value, path, context)
→ type 被 RegistryKeySchema 规范化
→ params 只检查字段存在且为 None 或 dict
→ 返回公共结构 dict
```

第二阶段由 `normalize_polymorphic_nodes()`：

```text
context.get_selection(path)
→ selection.spec.normalize_params(params, path.child("params"), context)
→ 把结果写回新的配置
```

为什么分两阶段：普通 Schema 知道公共 `{type, params}` 结构，但只有 Bootstrap 选择出的 ComponentSpec 知道具体参数字段和默认值。

## 6.8 `PathSchema.normalize()`

本质：路径值的结构节点，不是路径解析器。

第一阶段输入输出：

```text
str → pathlib.Path（可能仍是相对路径）
pathlib.Path → 原样或复制返回
None → allow_none 时返回 None
其他类型 → TypeMismatchError
```

后续 `resolve_paths()` 才做：

```text
选择基准
拼接相对路径
转成绝对路径
调用 deriver
```

再后续 Validator 才做：

```text
存在性
is_file/is_dir
后缀
非空
```

## 6.9 Mode 可选分区如何实现

不需要额外实现一个含糊的 `OptionalSectionSchema` 类。分区是否 required/optional/forbidden 直接通过 ModeSpec 的 Schema contribution 表达：

```text
required
→ Mode contribution 中加入 required=True 的节点

optional
→ 加入 required=False 的节点

forbidden
→ 不加入最终 Schema
→ Root Mapping 的未知字段规则会拒绝它
```

例如 Test Mode 不合并 `optimization`，用户写了该分区时，Root Mapping 直接报 UnknownFieldError。这样无需另一套分区状态机制。

---

## 7. Schema 合并与编译协议

最终 Schema 不是手写的完整任务模板，而是把本次运行实际选中的贡献组合起来：

```text
ROOT_SCHEMA
+ ModeSpec.schema_contribution
+ TaskSpec.schema_contribution
+ ComponentSpec.schema_contribution
```

## 7.1 `merge_schema()` 如何工作

签名：

```text
merge_schema(left, right, path, left_source, right_source) → SchemaNode
```

输入是两个描述同一 YAML 路径的 SchemaNode；返回一个新节点，不修改输入。

递归规则：

```text
Mapping + Mapping
→ 合并 fields
→ 同名字段递归 merge_schema
→ 独有字段复制到结果

Sequence + Sequence
→ 合并 item_schema
→ required/default/min/max/allow_none 必须兼容
→ default 都存在但值不同，抛 SchemaConflictError

Scalar + Scalar
→ python_types、required、allow_none、default、normalizer 必须兼容
→ default 的兼容性按 MISSING/实际值判断，值不同抛 SchemaConflictError

不同节点类型
→ SchemaConflictError
```

“后者覆盖前者”被禁止。若两个来源对同一路径给出不同默认值，系统无法判断谁正确，必须在注册时修复。

## 7.2 `collect_specs()` 先决定合并谁

```text
collect_specs(context, services) → SpecBundle
```

执行：

```text
services.root_schema
→ services.mode_registry.get(context.mode_key)
→ services.task_registry.get(context.task_key)
→ 遍历 context.selections 取得 component specs
→ 去重并稳定排序
→ 返回 SpecBundle
```

`SpecBundle` 只是输入集合，不是最终 Schema。

## 7.3 `compile_schema()` 执行合并和收集

```text
compile_schema(bundle) → CompiledSchema
```

执行：

```text
1. 从 Root Schema 开始
2. merge Mode contribution
3. merge Task contribution
4. merge 每个 Component contribution
5. walk_schema(final_root)
6. 收集所有 PathSchema
7. 收集所有 PolymorphicSchema
8. 收集 ValidatorRef
9. 记录每个 ConfigPath 的 source
10. 返回 CompiledSchema
```

返回数据：

```text
root
path_nodes
polymorphic_nodes
validator_refs
schema_sources
```

为什么需要编译结果：后续阶段必须使用同一份最终 Schema。若 Normalize、Path、Validator 各自重新组合，会出现规则不一致和重复查找。

---

## 8. Spec 与 Registry 如何协作

Spec 是声明对象，不是运行时组件实例。Registry 保存 Spec，Bootstrap 根据 YAML 的 `type` 选择 Spec，后续阶段调用 Spec 中的 Schema、Normalizer、Validator 或 Factory。

## 8.1 `ComponentSpec`

建议实现为不可变 dataclass。字段：

```text
category
key
aliases
params_schema
default_params
capabilities
normalizers
validators
factory
schema_contribution
```

创建者：`builtins/components/*.py` 中的 `build_xxx_spec()`。

调用过程：

```text
Registry.register(component_spec)
→ component_spec.validate_definition()

Bootstrap 读取 type
→ ComponentRegistry.get(category, type)
→ 创建 ComponentSelection(spec=component_spec)

normalize_polymorphic_nodes
→ component_spec.normalize_params(...)

validate_config
→ component_spec validators

Builder
→ component_spec.build(...)
```

`normalize_params()` 返回最终 dict；`validate_definition()` 和 Validator 成功返回 `None`；`build()` 返回真正的组件实例。

## 8.2 `TaskSpec`

建议实现为不可变 dataclass。字段：

```text
key
aliases
supported_modes
task_params_schema
schema_contribution
bootstrap_descriptors
required_capabilities_by_mode
normalizers
validators
cross_validators
builders_by_mode
```

创建者：`builtins/tasks/*.py`。

消费者和用途：

```text
Bootstrap
→ supports_mode(mode_key)
→ 取得 bootstrap_descriptors

Compile
→ 合并 schema_contribution

Validator Pipeline
→ 执行 validators/cross_validators

Runner
→ get_builder(mode_key)
```

`get_builder()` 执行映射查询：

```text
builders_by_mode[mode_key]
```

不拼接 `task + mode` 字符串。

能力要求示例：

```text
qa_sft:
  tokenizer → offset_mapping
  model → extractive_qa
  dataset → extractive_qa
  collator → qa_span_batching
```

## 8.3 `ModeSpec`

建议实现为不可变 dataclass。字段：

```text
key
aliases
schema_contribution
bootstrap_descriptors
required_sections
forbidden_sections
normalizers
validators
runner
```

调用过程：

```text
Bootstrap
→ ModeRegistry.get(mode_key)
→ 取得描述符和模式支持规则

Compile
→ 合并模式 Schema

Validator Pipeline
→ 执行模式 Validator

入口
→ mode_spec.get_runner()
```

`runner` 返回/引用一个可调用对象，实际运行时由入口调用。ModeSpec 自身不执行训练。

## 8.4 Registry 方法的实际行为

```text
register(spec) → None
```

执行 `spec.validate_definition()`，检查 key/alias 冲突，然后写入内部 dict。

```text
resolve_key(raw_key) → str
```

执行 `strip/casefold` 和 alias 查询，返回 canonical key；未知时抛 `UnknownRegistryKeyError`。

```text
get(key) → Spec
```

先解析 canonical key，再返回对应 Spec。

```text
contains(key) → bool
all_keys() → tuple[str, ...]
```

不同组件类别使用独立子 Registry，避免同名 Model 和 Dataset 冲突。Registry 不保存当前配置值，也不创建运行时组件。

---

## 9. Bootstrap 阶段

Bootstrap 解决的问题是：完整 Schema 依赖 `type`，但此时配置还没有完整规范化。因此它只做受限读取和 Registry 选择。

## 9.1 输入和返回

```text
bootstrap_config(
    raw_config,
    config_path,
    mode_registry,
    task_registry,
    component_registry,
    expected_mode=None,
    validate_runtime=True,
) → ParseContext
```

输入 `raw_config` 是 `yaml.safe_load()` 返回的原始 dict；返回的 `ParseContext` 不包含修改后的配置。

## 9.2 具体执行

```text
1. 检查 raw_config 是 Mapping
2. 从 raw_config["run"]["mode"] 读取字符串
3. mode_registry.resolve_key → mode_key
4. expected_mode 非 None 时也规范化并比较
5. 从 raw_config["task"]["type"] 读取字符串
6. task_registry.resolve_key → task_key
7. task_spec = task_registry.get(task_key)
8. task_spec.supports_mode(mode_key)，否则报错
9. 取得 ModeSpec/TaskSpec.bootstrap_descriptors
10. 每个 descriptor 扫描一个或多个 YAML 节点
11. 从节点 Mapping 中读取 type
12. component_registry.get(category, raw_type) → ComponentSpec
13. 为真实路径创建 ComponentSelection
14. 保存到 selections[ConfigPath]
15. 返回 ParseContext
```

例如 Loss descriptor 扫描：

```text
raw_config["components"]["losses"]
→ enumerate 列表
→ 第 0 项路径 config.components.losses[0]
→ 读取 type="CrossEntropyLoss"
→ Registry 返回 canonical key="cross_entropy"
→ 创建 ComponentSelection
```

Bootstrap 不修改 YAML 中的原始 `type`；后续 `RegistryKeySchema.normalize()` 才在返回配置中写入 canonical key。

## 9.3 描述符从哪里来

公共的 `run.mode` 和 `task.type` 由 Bootstrap 固定读取，因为它们决定整个解析过程。

其他动态位置由 ModeSpec/TaskSpec 提供，例如：

```text
components.model
components.losses[*]
evaluation.metrics.*[*]
```

Bootstrap 引擎只解释 descriptor，不知道这些节点属于 MLM、QA 还是 GPT。

## 9.4 本阶段明确不做

```text
不补默认值
不调用 SchemaNode.normalize
不合并 params
不解析路径
不检查数值范围
不检查文件系统
不构造组件
```

这些职责若进入 Bootstrap，会导致后续 Schema 流程重复或顺序混乱。

---

## 10. 从 ParseContext 到 CompiledSchema

这一阶段没有读取或修改配置值，只组合规则。

### `collect_specs(context, services) → SpecBundle`

具体查询：

```text
root_schema = services.root_schema
mode_spec = services.mode_registry.get(context.mode_key)
task_spec = services.task_registry.get(context.task_key)
component_specs = tuple(selection.spec for selection in context.selections.values())
```

组件 Spec 要去重并按 `(category, key, path)` 稳定排序。返回 `SpecBundle` 数据对象。

### `compile_schema(bundle) → CompiledSchema`

使用第 7 章的 `merge_schema()` 逐个合并，然后 `walk_schema()` 收集元数据。成功返回 CompiledSchema，冲突时抛 SchemaConflictError。

该阶段不调用任何 `normalize()`，因为现在只处理 Schema 对象，不处理 YAML 值。

---

## 11. `normalize_structure()` 的实际递归

接口：

```text
normalize_structure(raw_config, compiled_schema, context) → dict
```

实现核心只有一次根调用：

```text
result = compiled_schema.root.normalize(
    raw_config,
    ConfigPath.root("config"),
    context,
)
```

根 `MappingSchema.normalize()` 再逐层调用子节点，因此不需要 `normalize_structure()` 自己写一套字段递归。

示例：

```text
Root Mapping
→ experiment 子 Mapping
→ seed 子 Scalar
→ seed 缺失，Scalar 返回默认值 42
→ 各父 Mapping 把返回值写入新 dict
→ 最终返回完整新配置
```

阶段结束时：

```text
未知字段已拒绝
必填字段已确认
普通默认值已填充
None 已检查
list/dict 已复制
Registry key 已 canonicalize
Path 已转 pathlib.Path，但可能是相对路径
Polymorphic params 仍可能是 None 或用户 dict
```

输入 `raw_config` 不得变化。

---

## 12. `normalize_polymorphic_nodes()` 如何处理 params

接口：

```text
normalize_polymorphic_nodes(config, compiled_schema, context) → dict
```

这不是再次递归全部字段，只遍历 `compiled_schema.polymorphic_nodes`。

每个节点执行：

```text
1. node_config = get_value(config, node_path)
2. selection = context.get_selection(node_path)
3. 检查 node_config["type"] == selection.key
4. raw_params = node_config["params"]
5. final_params = selection.spec.normalize_params(
       raw_params,
       node_path.child("params"),
       context,
   )
6. 把 final_params 写入新配置副本
```

`ComponentSpec.normalize_params()` 内部执行：

```text
result = deepcopy(default_params)
→ raw_params is None 时不覆盖
→ raw_params 是 dict 时 result.update(raw_params)
→ params_schema.normalize(result, params_path, context)
→ 依次调用 normalizers
→ 确认并返回 dict
```

无参数组件：

```text
default_params={}
params_schema=空 MappingSchema
```

因此 `null` 和 `{}` 最终都返回 `{}`。

第一版不自动发现 params 内的未知嵌套判别字段；需要时必须显式登记嵌套 PolymorphicSchema 和 BootstrapDescriptor。

---

## 13. Override 阶段如何执行

CLI 不直接修改 YAML dict，而是先创建 `ConfigOverride`。

```text
apply_overrides(config, overrides, compiled_schema, context) → dict
```

对每个 Override：

```text
1. 根据 ConfigPath 在 CompiledSchema 中找到 SchemaEntry
2. 找不到 → UnknownFieldError
3. 若路径是 run.mode、task.type 或 component.type → 拒绝
4. normalized_value = target_node.normalize(
       override.raw_value,
       override.path,
       context,
   )
5. set_value_copy 把值写入配置副本
6. 处理下一项
7. 返回新 dict
```

为什么再次调用 `normalize()`：CLI 传入的是原始值，也必须遵守目标节点的类型、None 和基础结构规则。

`--resume` 由入口转换为：

```text
ConfigOverride(
    path=config.run.resume_from,
    raw_value=命令行字符串,
    source="cli:--resume",
)
```

---

## 14. Schema 驱动路径解析

```text
resolve_paths(config, compiled_schema, context, deriver_registry) → dict
```

它只遍历 `compiled_schema.path_nodes`，不按字段名搜索配置。

### 路径基准

```text
config_dir
→ 使用 context.config_dir

field_reference(data.dataset.root)
→ 使用配置中该字段已经解析后的 Path
```

### 为什么需要依赖排序

例如 Split 路径依赖 Dataset Root：

```text
config.data.dataset.root
→ 先解析

config.data.dataset.splits.train
→ 再以 root 为基准解析
```

实现：

```text
PathNode 列表
→ 建 dependency 图
→ graphlib.TopologicalSorter
→ 得到解析顺序
```

### 每个路径值如何处理

```text
当前值 None
├── 有 deriver → 调用命名 deriver 返回 Path
├── allow_none → 保留 None
└── 否则 → PathResolutionError

当前值不是 str/pathlib.Path（且不是允许的 None）
→ TypeMismatchError

当前值是相对 Path
├── base=config_dir → context.config_dir / value
└── base=字段引用 → resolved_base / value

当前值是绝对 Path
→ 直接 resolve(strict=False)

字段引用不存在或尚未按依赖顺序解析
→ PathDependencyError

deriver 不存在、不可调用或返回值无法转换为 Path
→ PathResolutionError
→ 若属于 deriver 注册/定义错误，抛 TypeError
```

把结果写入新配置。这里不检查 `exists/is_file/is_dir`；这些是 Validator 的工作。

---

## 15. 注册式校验流水线

详细字段规则见 `validate_config_guide.md`。本节只说明“谁被调用、什么时候调用、返回什么”。

### `ValidatorSpec`

一个 ValidatorSpec 是一个不可变数据对象，不是校验结果。

```text
name
phase
priority
applies_to(context, config)
validate(context, config)
source
```

调用：

```text
should_run(context, config) → bool
run(context, config) → None
```

当 `should_run()` 为 True 时调用 `validate()`；成功必须返回 None，失败直接抛 ConfigValidationError 或其子类。

### `collect_validators()`

```text
collect_validators(compiled_schema, context) → tuple[ValidatorSpec, ...]
```

来源：

```text
Schema 节点
ComponentSpec
ModeSpec
TaskSpec
Runtime
```

按 `phase_order, priority, name, source` 稳定排序，返回不可变 tuple。

### `validate_config()`

```text
validate_config(config, context, compiled_schema) → None
```

执行：

```text
收集 Validator
→ 依次 should_run
→ true 时 run
→ 所有成功后返回 None
```

它不修改 config、不补默认值、不解析路径、不构造组件，也不捕获并吞掉 ConfigError。

Runtime Validator 只有在 `context.validate_runtime=True` 时执行；`runtime.py` 才允许导入 torch。

---

## 16. 配置快照与返回契约

### `ParsedConfig`

`load_config()` 成功返回 `ParsedConfig` 数据对象，而不是只返回 dict：

```text
config: dict
context: ParseContext
compiled_schema: CompiledSchema
schema_fingerprint: str
```

Builder 主要使用 `parsed.config`；调试、复现和测试可以使用其余字段。

`config` 的不变量：

```text
Registry key 是 canonical key
普通默认值已填充
params 都是 dict
非 None 路径都是绝对 pathlib.Path
不包含 Schema、Spec 或运行时实例
```

### 快照函数

```text
to_serializable_config(config) → dict
save_config_snapshot(config, output_path) → pathlib.Path
```

`to_serializable_config()` 递归复制配置：

```text
Path → str
Mapping → 新 dict
list → 新 list
YAML 标量 → 原样
其他对象 → TypeError
```

`save_config_snapshot()` 使用 `yaml.safe_dump(..., allow_unicode=True)`，保存最终生效配置。建议同时保存：

```text
raw_config.yaml
resolved_config.yaml
```

原始快照用于查看用户输入，resolved 快照用于复现实际运行。

---

## 17. Builder 新契约

### Builder 如何选择

```text
task_spec = services.task_registry.get(parsed.context.task_key)
builder = task_spec.get_builder(parsed.context.mode_key)
```

`get_builder()` 返回函数或其他可调用对象，不返回 Builder 名称字符串。找不到对应 mode 或映射值不可调用时，抛 `ModeTaskMismatchError` 或 `TypeError`；前者表示任务不支持该模式，后者表示 TaskSpec 定义错误。

### `BuildContext`

这是 Runner 创建的数据对象：

```text
parsed_config
services
logger
experiment_paths
device_summary
```

它不负责构造组件，只把 Builder 所需的运行依赖集中传入。

### `build_component()`

```text
build_component(
    category,
    node_config,
    build_context,
    dependencies=None,
) → Any
```

执行：

```text
1. type_key = node_config["type"]
2. spec = services.component_registry.get(category, type_key)
3. params = node_config["params"]
4. 将 Tokenizer、Optimizer 等运行依赖显式加入调用参数
5. return spec.build(params, build_context/dependencies)
```

返回值是真实组件实例，例如 Tokenizer、Model 或 Optimizer。

Task Builder 负责依赖连接：

```text
Tokenizer 实例 → Dataset/Collator
Tokenizer.vocab_size/pad_token_id → Model
Task.parameters() → Optimizer
Optimizer → Scheduler
```

### Train Builder 的典型执行

```text
build tokenizer
→ build datasets
→ build collator
→ 创建 DataLoaders
→ build model
→ build losses
→ 创建 Task
→ build metrics
→ build optimizer
→ build scheduler
→ 创建 Trainer
→ 返回 TrainBuildResult
```

每一步都消费前面返回的对象，不只是按名称排列。

### Test/Inference 的训练配置

```text
current_parsed = load_config(test_or_inference_yaml)
→ train_config_path = current_parsed.config["run"]["checkpoint"]["train_config"]
→ train_parsed = load_config(train_config_path, expected_mode="train", validate_runtime=False)
→ 检查两者 task.type 一致
→ 用 train_parsed 构造 Tokenizer/Model/Task
→ 用 current_parsed 构造 Dataset/Metrics/Device/Decoding
→ 加载 checkpoint.path 权重
```

不允许当前 Test/Inference YAML 覆盖模型结构参数。

---

## 18. 建议文件结构

```text
main/parse/
├── api.py
├── context.py
├── errors.py
├── bootstrap.py
├── pipeline.py
├── overrides.py
├── snapshot.py
│
├── schema/
│   ├── nodes.py
│   ├── merge.py
│   ├── compile.py
│   ├── normalize.py
│   └── walk.py
│
├── specs/
│   ├── component.py
│   ├── task.py
│   ├── mode.py
│   └── validator.py
│
├── registries/
│   ├── base.py
│   ├── components.py
│   ├── tasks.py
│   └── modes.py
│
├── paths/
│   ├── graph.py
│   ├── resolver.py
│   └── derivers.py
│
├── validation/
│   ├── pipeline.py
│   ├── common.py
│   └── runtime.py
│
├── builtins/
│   ├── root_schema.py
│   ├── modes/
│   ├── tasks/
│   └── components/
│
├── parser_refactor_plan.md
└── validate_config_guide.md
```

实现中允许先少量文件，后续再拆；但模块职责必须保持。

---

## 19. 公共 API

## 19.1 `load_config`

```text
load_config(
    yaml_path,
    *,
    expected_mode=None,
    overrides=None,
    validate_runtime=True,
) → ParsedConfig
```

完整流程：

```text
load_raw_yaml
→ bootstrap_config
→ collect_specs
→ compile_schema
→ normalize_structure
→ normalize_polymorphic_nodes
→ apply_overrides
→ resolve_paths
→ validate_config
→ fingerprint_schema
→ ParsedConfig
```

## 19.2 入口接口

```text
parse_train_args() → args
parse_test_args() → args
parse_inference_args() → args
```

三者只定义真实使用的 CLI 参数。

## 19.3 注册接口

```text
register_mode(spec)
register_task(spec)
register_component(spec)
```

注册在程序启动、配置加载之前完成。

---

## 20. 内置 Schema 贡献

本节不是重新定义固定的任务总 Schema，而是说明内置 Spec 如何贡献字段。实现时应先完成 Root/Mode 的公共结构，再由 Task 和 Component 动态增加字段。

## 20.1 Root Schema

固定：

```text
version
experiment
run
task
```

其余分区由 ModeSpec 增加。

## 20.2 Train ModeSpec

增加：

```text
components
data
optimization
evaluation
runtime
```

声明：

```text
run.resume_from 可选
run.checkpoint 禁止
```

## 20.3 Test ModeSpec

增加：

```text
run.checkpoint
components.losses
data
evaluation.metrics.test
runtime
```

禁止：

```text
optimization
inference
当前 YAML 中的 model/tokenizer
```

## 20.4 Inference ModeSpec

增加：

```text
run.checkpoint
inference
runtime
```

禁止：

```text
optimization
evaluation
data
components
```

## 20.5 TaskSpec 贡献

TaskSpec 只增加：

```text
task.params Schema
能力要求
任务 Validator
Builder
可能的任务专属分区字段
```

TaskSpec 不复制完整 Train/Test Schema。

---

## 21. MLM 和 QA 作为内置扩展示例

## 21.1 MLM

注册：

```text
TaskSpec: mlm_pretrain
Tokenizer: wordpiece
Model: tinybert_mlm
Dataset: zhwiki_mlm
Collator: dynamic_mlm
Loss: cross_entropy
Metric: token_accuracy
```

能力：

```text
model.masked_lm
dataset.token_sequence
collator.dynamic_mlm
```

## 21.2 QA

注册：

```text
TaskSpec: qa_sft
Tokenizer: qa_wordpiece
Model: tinybert_qa
Dataset: cmrc2018
Collator: qa_span
Metric: exact_match / character_f1
```

能力：

```text
tokenizer.offset_mapping
model.extractive_qa
dataset.extractive_qa
collator.qa_span_batching
```

## 21.3 GPT 扩展验证

未来 GPT 接入只新增：

```text
GPT ComponentSpecs
GPT TaskSpec
GPT Validators
GPT Builders
GPT YAML
```

不得修改：

```text
bootstrap.py
schema/merge.py
schema/normalize.py
paths/resolver.py
validation/pipeline.py
api.load_config
```

如果必须修改，说明协议不够通用，需要先讨论抽象缺口。

---

## 22. 错误类型

建议定义：

```text
ConfigError
YamlLoadError
BootstrapError
UnknownRegistryKeyError
ModeTaskMismatchError
SchemaError
SchemaConflictError
MissingFieldError
UnknownFieldError
TypeMismatchError
NullNotAllowedError
ParameterNormalizationError
PathResolutionError
PathDependencyError
ConfigValidationError
RuntimeValidationError
```

每个异常保存：

```text
path
message
actual_value
expected
source
```

最终入口负责统一格式化，不在底层错误文本中加入口语后缀。

---

## 23. 详细实施顺序

改动幅度很大，必须按基础协议到垂直链路逐步推进。

## 23.1 阶段零：冻结旧系统

操作：

```text
保留 parse.copy
保留当前入口引用
记录现有五类 YAML
记录各 Builder 字段消费路径
不在旧 Parser 上继续增加功能
```

完成标准：

```text
旧链路仍可作为行为对照
新代码未接管入口
```

## 23.2 阶段一：错误路径和 Schema 基础节点

实现顺序：

```text
MISSING
ConfigPath
ConfigError
ScalarSchema
MappingSchema
SequenceSchema
EnumSchema
RegistryKeySchema
PathSchema
PolymorphicSchema
```

先只做节点构造和独立规范化，不接 Registry。

测试重点：

```text
输入不被修改
缺失与 None 区分
默认可变对象深拷贝
未知字段
列表索引路径
```

## 23.3 阶段二：Schema Merge 与 Compile

实现：

```text
merge_schema
SchemaConflictError
compile_schema
schema walk
Path/Polymorphic/Validator 收集
```

先用人工小 Schema 测试冲突，不接真实任务。

## 23.4 阶段三：Spec 与 Registry

实现：

```text
BaseRegistry
ComponentSpec
TaskSpec
ModeSpec
ValidatorSpec
各类 Registry
```

注册时立即验证 Spec 自身。

第一批只注册：

```text
train/test/inference ModeSpec
qa_sft TaskSpec
QA Test 所需 ComponentSpec
```

## 23.5 阶段四：Bootstrap

实现：

```text
BootstrapDescriptor
bootstrap_config
ComponentSelection
ParseContext
```

先验证：

```text
mode/task 解析
别名解析
mode/task 支持关系
动态列表 type 扫描
未知组件
```

## 23.6 阶段五：结构与 params 规范化

实现：

```text
normalize_structure
normalize_polymorphic_nodes
```

重点验证：

```text
params=None
params={}
params=部分参数
未知参数
无参数组件
最终 params=dict
```

## 23.7 阶段六：Override

实现：

```text
ConfigOverride
apply_overrides
--resume 转换
```

禁止覆盖判别字段。

## 23.8 阶段七：路径依赖图

实现：

```text
收集 PathSchema
建立依赖
拓扑排序
解析 config_dir 基准
解析 field reference 基准
train_config deriver
```

重点验证相对路径和循环依赖。

## 23.9 阶段八：Validator Pipeline

实现：

```text
validator 收集
稳定排序
阶段执行
validate_runtime 开关
```

先接 QA Test 所需规则。

## 23.10 阶段九：QA Test 垂直切片

完整链路：

```text
新 QA Test YAML
→ load_config(expected_mode=test)
→ 加载训练配置 expected_mode=train
→ 新 QA Test Builder
→ 构造组件但不运行完整测试
```

这是第一次改 Builder 的阶段。

完成标准：

```text
旧 MLM 字段不再要求
EM/F1 params=None 正确变成 {}
Checkpoint 与训练配置可加载
Builder 不访问旧扁平路径
```

## 23.11 阶段十：QA Train

注册并实现 QA Train 所需全部 Specs。迁移：

```text
start_train
QA Builder
实验配置快照
resume/init_from 互斥
```

先做 Parser 和 Builder 构造，再做最小 Forward。

## 23.12 阶段十一：MLM Test/Train

新增 MLM Specs，不修改 Parser 引擎。若发现必须改引擎，暂停并讨论协议缺口。

## 23.13 阶段十二：MLM Inference

新增 Decoding ComponentSpec 和 Inference Builder。

## 23.14 阶段十三：入口切换

逐个切换：

```text
start_test
start_train
start_inference
```

每切一个都保留可回退的旧入口引用，确认后再继续。

## 23.15 阶段十四：清理旧系统

仅在所有正式链路通过后删除：

```text
parse.copy
旧 CUSTOM_DICT
旧重复 Validator
旧模式化任务 Registry 键
旧 YAML
```

---

## 24. 每阶段验证层级

必须依次区分：

```text
Schema 单元测试
Registry 单元测试
Bootstrap 测试
Normalize 测试
Path 测试
Validator 测试
Parser 组合测试
Builder 构造测试
单 Batch 测试
完整运行测试
```

前一级未通过，不进入后一级。

不能用“YAML 可以加载”代表 Parser 完成，也不能用“Builder 可以构造”代表训练通过。

---

## 25. 关键测试矩阵

### Schema

```text
MISSING / None / 默认值
未知字段
错误类型
Sequence 空列表
Schema 冲突
默认 dict 深拷贝
```

### Registry

```text
重复 key
别名冲突
未知 key
规范键输出
```

### Bootstrap

```text
错误 mode
错误 task
mode/task 不匹配
列表组件扫描
不支持组合
```

### Params

```text
None → defaults
{} → defaults
部分参数 → defaults + override
未知参数 → error
无参数组件 → {}
```

### Paths

```text
config_dir
字段基准
推导路径
绝对路径
循环依赖
不存在文件
```

### Mode

```text
Train 禁止 checkpoint
Test 禁止 optimization
Inference 禁止 data/evaluation
```

### Task capability

```text
QA Tokenizer 缺少 offset_mapping
MLM Collator 缺少 dynamic_mlm
GPT 模拟组件可在不改 Parser 的情况下注册
```

---

## 26. 实现时的决策原则

每增加一项规则先回答：

```text
它是 Parser 引擎规则，还是扩展规则？
它属于 Schema、Normalizer、Path Resolver 还是 Validator？
它由 Mode、Task 还是 Component 拥有？
新增另一个任务时是否需要修改引擎？
它的消费者是谁？
它的错误路径是什么？
它是否修改配置？
它在流水线哪个阶段执行？
```

如果规则中出现具体任务或组件名称，优先放入对应 Spec，而不是核心流程。

---

## 27. 最终验收标准

### 引擎通用性

```text
Parser 核心不知道 MLM、QA、GPT
新增任务不修改通用流水线
新增组件不修改普通递归器
路径解析不按字段名硬编码
Validator 总调度器不含任务分支
```

### 配置正确性

```text
mode/task 明确分离
所有 Registry key 已规范化
所有 params 最终为 dict
所有路径最终为绝对 Path
原始配置不被修改
最终配置可以生成可复现快照
```

### 架构完整性

```text
Schema 冲突可诊断
Spec 注册错误可诊断
路径依赖可诊断
所有错误包含 ConfigPath
Test/Inference 训练配置来源明确
Builder 使用新配置契约
```

### 扩展性证明

在不修改 Parser 引擎的情况下，注册一个最小模拟任务和模拟组件并成功完成：

```text
Bootstrap
→ Schema Compose
→ Normalize
→ Path
→ Validate
```

未来 GPT 接入时也应遵守同一证明标准。

---

# 三十、逐文件实现手册

本章将前面的架构转换为可以直接执行的实现计划。后续实际编写时，优先按照本章顺序创建文件和接口，不要先修改入口或 Builder。本文只规定实现方式，不提供完整代码。

## 30.1 依赖边界

### Python 标准库

Parser 核心只使用标准库即可完成：

```text
abc                 → 抽象接口和协议基类
copy                → deepcopy 默认值和配置副本
 dataclasses        → 不可变或结构化上下文对象
 enum               → 节点类型、校验阶段等有限枚举
 pathlib            → 路径对象和路径解析
 re                  → 必要的路径/设备格式检查
 math                → 有限数值检查
 collections.abc     → Mapping、Sequence 等容器判断
 typing              → 类型标注
 heapq               → 可选的稳定优先队列
 graphlib            → Python 3.9+ 的 TopologicalSorter
 hashlib             → Schema/config fingerprint
 argparse            → CLI 参数
 json                → 快照辅助序列化
```

实际项目中不需要为了 Parser 引入新的第三方 Schema 框架。自行实现节点协议更适合本项目教学目标，也能让错误路径和各阶段行为完全透明。

### 现有第三方库

```text
PyYAML              → yaml.safe_load 读取 YAML
torch               → Runtime Validator 检查 CUDA；不用于 Schema 核心
```

`yaml.safe_load()` 只能负责 YAML 到 Python 对象的转换。它不能替代项目的 Schema 校验，也不能判断任务、组件和路径语义。

### 禁止的依赖方向

```text
schema 节点不能导入 Builder
Parser 不能导入具体 MLM/QA Task
通用 normalize 不能导入 torch 模型
路径解析不能导入 Dataset/Checkpoint Builder
Validator 不能创建 Optimizer 或 Model
```

这样才能保证未来增加 GPT 时不会把 GPT 代码反向耦合进 Parser 引擎。

## 30.2 文件创建顺序

第一轮实际创建顺序固定为：

```text
1. main/parse/errors.py
2. main/parse/context.py
3. main/parse/schema/nodes.py
4. main/parse/schema/merge.py
5. main/parse/schema/normalize.py
6. main/parse/specs/component.py
7. main/parse/specs/mode.py
8. main/parse/specs/task.py
9. main/parse/registries/base.py
10. main/parse/registries/components.py
11. main/parse/registries/modes.py
12. main/parse/registries/tasks.py
13. main/parse/bootstrap.py
14. main/parse/schema/compile.py
15. main/parse/paths/graph.py
16. main/parse/paths/resolver.py
17. main/parse/validation/pipeline.py
18. main/parse/api.py
19. main/parse/builtins/root_schema.py
20. main/parse/builtins/modes/
21. main/parse/builtins/tasks/
22. main/parse/builtins/components/
```

先创建基础协议，再创建内置扩展，最后创建入口适配。原因是依赖方向必须保持：

```text
基础协议
→ 注册机制
→ 内置 Spec
→ 通用流水线
→ 兼容入口
→ Builder
```

## 30.3 阶段一：`errors.py`

文件：

```text
main/parse/errors.py
```

当前磁盘中的 `errors.py` 为空，而且 `ConfigPath` 尚未实现。因此本阶段只建立异常继承关系，不提前依赖 `context.py`，也不实现结构化路径字段。

### 本阶段究竟实现什么

本文件需要实现一组类，不实现普通函数。第一版中所有类都可以只有类声明和 `pass`，不需要分别编写构造函数。

继承关系：

```text
Exception
└── ConfigError
    ├── YamlLoadError
    ├── BootstrapError
    │   ├── UnknownRegistryKeyError
    │   └── ModeTaskMismatchError
    ├── SchemaError
    │   ├── SchemaConflictError
    │   ├── MissingFieldError
    │   ├── UnknownFieldError
    │   ├── TypeMismatchError
    │   └── NullNotAllowedError
    ├── ParameterNormalizationError
    ├── PathResolutionError
    │   └── PathDependencyError
    ├── ConfigValidationError
    └── RuntimeValidationError
```

这里的子类和 `ConfigPath` 完全不是同一种东西：

```text
ConfigPath
→ 普通数据类
→ 表示配置中的位置
→ 以后放在 context.py

ConfigError 及其子类
→ Python 异常类
→ 表示不同种类的失败
→ 放在 errors.py
```

因此不能“像 ConfigPath 一样写”。`ConfigPath` 后面会有 `child()`、`index()` 等路径方法；异常子类第一版不需要任何方法。

### `ConfigError`

第一版职责：作为所有 Parser 异常的统一父类，使调用方可以一次捕获所有配置错误。

第一版不需要自定义 `__init__()`。因为它继承 `Exception` 后，天然可以接收任意异常消息参数：

```text
ConfigError("配置错误")
MissingFieldError("缺少 task.type")
```

Python 的 `Exception` 会保存传入参数，因此：

```text
str(error)
→ 得到错误消息

error.args
→ 得到构造时传入的参数元组
```

如果开发者希望显式写构造函数，第一版也只应接收 `message` 并调用父类；但这不是必需步骤。推荐先不重写构造函数，减少无意义代码。

### 每个子类怎么写

每个子类第一版只负责提供一个明确的异常类型，不保存额外字段，也不实现额外方法。

概念结构：

```text
YamlLoadError(ConfigError)
→ YAML 文件读取或解析失败

BootstrapError(ConfigError)
→ Bootstrap 阶段失败

UnknownRegistryKeyError(BootstrapError)
→ mode、task 或 component type 未注册

ModeTaskMismatchError(BootstrapError)
→ Task 不支持当前 Mode

SchemaError(ConfigError)
→ Schema 构造或结构规范化错误

SchemaConflictError(SchemaError)
→ 两份 Schema 对同一路径声明冲突

MissingFieldError(SchemaError)
→ 缺少必填字段

UnknownFieldError(SchemaError)
→ 出现 Schema 未允许的字段

TypeMismatchError(SchemaError)
→ Python 类型不符合 Schema

NullNotAllowedError(SchemaError)
→ 用户显式写 null，但该节点不允许 None

ParameterNormalizationError(ConfigError)
→ 多态 params 合并或规范化失败

PathResolutionError(ConfigError)
→ 路径基准、转换或推导失败

PathDependencyError(PathResolutionError)
→ 路径依赖图存在缺失依赖或环

ConfigValidationError(ConfigError)
→ 结构正确，但业务值或跨字段关系非法

RuntimeValidationError(ConfigError)
→ 配置本身正确，但当前 CUDA 等运行环境不满足
```

### 为什么需要这么多空子类

它们不是为了分别保存不同数据，而是让调用方能通过异常类型判断失败阶段。

例如：

```text
except YamlLoadError
→ 提示用户检查 YAML 文件

except SchemaError
→ 提示配置结构不符合 Schema

except RuntimeValidationError
→ 提示切换设备或检查环境

except ConfigError
→ 捕获所有 Parser 异常
```

如果全部使用 `ValueError`，上层只能解析错误字符串，无法稳定区分错误来源。

### 本阶段的使用方式

发现错误的位置直接抛出对应子类：

```text
缺少字段
→ raise MissingFieldError("缺少必填字段：config.task.type")

未知组件
→ raise UnknownRegistryKeyError("未注册的模型类型：unknown_model")

路径依赖环
→ raise PathDependencyError("路径依赖存在环：A → B → A")
```

构造异常对象后不返回它，也不在底层打印它，而是使用 `raise` 向上传递。

### 第一版完成标准

```text
errors.py 不导入 ConfigPath
errors.py 不导入 typing、dataclasses 或第三方库
ConfigError 直接继承 Exception
所有子类继承关系与上面的树一致
所有子类第一版可以只写 pass
每个子类都可以接收字符串消息并正常 raise
except ConfigError 可以捕获任意 Parser 子异常
except SchemaError 可以捕获所有 Schema 子异常
```

### 后续何时扩展

只有完成 `context.py` 中的 `ConfigPath`，并且 Schema/Bootstrap 已经开始实际抛错后，再回到 `errors.py` 扩展结构化字段：

```text
path
actual_value
expected
source
```

届时只需要扩展父类 `ConfigError.__init__()`，大多数子类仍然不需要自定义构造函数，因为它们会自动继承父类构造函数。

所以当前实现结论是：

```text
ConfigError
→ 继承 Exception
→ 第一版可以 pass

所有列出的子类
→ 按继承树继承 ConfigError 或中间父类
→ 第一版都可以 pass

ConfigPath
→ 暂时不实现、不导入、不参与 errors.py
```

为什么先做：后续所有模块可以立即使用明确异常类型；等 `ConfigPath` 实现后，只扩展父类即可，不需要重写全部子类。

## 30.4 阶段二：`context.py`

### 本阶段文件

```text
main/parse/context.py
```

### 本阶段依赖

```text
标准库：dataclasses、pathlib、typing
项目内：暂时只导入 errors.py 中必要的异常；不要导入 Schema、Spec、Builder
第三方库：无
```

### 本阶段实现结果

本文件实现：

```text
1 个缺失值哨兵：MISSING
3 个数据类：ConfigPath、ComponentSelection、ParseContext
```

这些对象只保存和传递状态，不读取 YAML、不访问 Registry、不规范化配置、不构造组件。

### `MISSING`：实现一个哨兵对象

实现形式：普通私有类加一个模块级单例，不是异常，也不是 dataclass。

```text
class _MissingType
模块级对象 MISSING = _MissingType()
```

建议方法：

```text
__repr__() → "MISSING"
```

不需要其他业务方法。

为什么需要：

```text
字段不存在 → MISSING
用户写 null → None
用户写值 → 具体值
```

函数返回时若要表达“没有默认值”，也使用 `MISSING`，不能使用 `None`。

完成检查：

```text
MISSING is MISSING
repr(MISSING) == "MISSING"
MISSING is not None
```

### `ConfigPath`：实现一个不可变数据类

实现形式：`@dataclass(frozen=True)`。

建议内部字段：

```text
parts: tuple[str | int, ...]
```

不要只保存一个拼好的字符串，否则 `child()` 和 `index()` 后续容易重复解析字符串。

需要实现的方法：

```text
ConfigPath.root(name: str) → ConfigPath
child(name: str) → ConfigPath
index(value: int) → ConfigPath
parent() → ConfigPath | None
__str__() → str
```

每个方法做什么：

```text
root
→ 创建根路径
→ 输入 "config"
→ 返回 parts=("config",)

child
→ 返回新对象
→ 原对象不变
→ 字符串字段使用点号

index
→ 返回新对象
→ int 部分显示为 [n]

parent
→ 根节点返回 None
→ 其他路径返回去掉最后一个 part 的新对象

__str__
→ parts=("config", "losses", 0, "params")
→ 返回 config.losses[0].params
```

参数检查：

```text
root/child 名称必须是非空 str
index 必须是非负严格 int
```

第一版参数错误可使用内置 `TypeError`/`ValueError`，也可使用 `ConfigError`；项目最终统一前不要让 ConfigPath 反向依赖 Schema。

为什么需要：Schema、Bootstrap、路径解析和 Validator 都需要追加路径。不可变对象可以防止某个递归分支修改其他分支共享的路径。

完成检查：

```text
原路径调用 child/index 后不变
字段和索引格式正确
可以作为 dict key
相同 parts 的对象相等
```

### `ComponentSelection`：实现一个不可变数据类

实现形式：`@dataclass(frozen=True)`。

字段：

```text
category: str
key: str
path: ConfigPath
spec: Any
```

当前为了避免 `context.py` 与 `specs/component.py` 循环导入，`spec` 第一版可以标注为 `Any`，后续使用 `TYPE_CHECKING` 做静态类型引用。

它不需要自定义方法。Bootstrap 创建它，Compile、参数规范化和 Builder 查询它。

为什么保存 `spec`：避免后续阶段再次使用 key 查询 Registry；Bootstrap 已经确认过的选择应成为稳定上下文。

### `ParseContext`：实现一个不可变数据类

实现形式：`@dataclass(frozen=True)`。

字段和类型：

```text
config_path: pathlib.Path
config_dir: pathlib.Path
expected_mode: str | None
mode_key: str
task_key: str
selections: Mapping[ConfigPath, ComponentSelection]
validate_runtime: bool
```

构造者：

```text
bootstrap_config()
```

消费者：

```text
collect_specs()
compile_schema()
normalize_polymorphic_nodes()
resolve_paths()
validate_config()
Builder/Runner
```

需要实现的方法：

```text
get_selection(path: ConfigPath) → ComponentSelection
```

这里的“存在”不是指 `ConfigPath` 对象自身具有某个 `exists()` 方法，也不是检查文件系统，更不是检查 YAML 节点是否存在。它指的是：

```text
传入的 ConfigPath 是否作为 key 存在于 ParseContext.selections 映射中
```

`ConfigPath` 只负责表示路径；`ParseContext.selections` 保存 Bootstrap 已识别出的“路径 → 组件选择”关系。

方法行为：

```text
path in self.selections
→ 返回 self.selections[path]

path not in self.selections
→ 抛 BootstrapError
```

推荐实现逻辑：

```text
get_selection(path):
    try:
        return self.selections[path]
    except KeyError as error:
        raise BootstrapError(
            f"未找到组件选择：{path}"
        ) from error
```

因此，调用者必须传入与 Bootstrap 保存时结构相同的 `ConfigPath`。例如：

```text
Bootstrap 保存：
config.components.model → ComponentSelection(...)

后续查询：
context.get_selection(
    ConfigPath.root("config").child("components").child("model")
)
→ 返回对应 ComponentSelection
```

之所以可以直接把 `ConfigPath` 用作 dict key，是因为它使用 `@dataclass(frozen=True)`，且内部 `parts` 是 tuple；具有相同 `parts` 的两个 `ConfigPath` 会相等并拥有一致的哈希值。

这里不要写：

```text
path.exists()
path.is_file()
```

这些方法属于文件系统路径检查，不属于 `ConfigPath`。文件存在性由 `PathSchema` 和 Validator 在后续阶段处理。

推荐抛项目异常，错误消息包含 `str(path)`。

ParseContext 不保存：

```text
raw_config
normalized_config
CompiledSchema
运行时组件实例
```

### 本阶段返回物

`context.py` 本身不运行流程。完成后，其他模块可以导入四个公共对象：

```text
MISSING
ConfigPath
ComponentSelection
ParseContext
```

### 本阶段完成标准

```text
只使用标准库
没有导入具体任务或组件
ConfigPath 不可变
ParseContext 不可变
ComponentSelection 不执行 Registry 查询
MISSING 与 None 明确区分
所有基础行为有独立最小测试
```

## 30.5 阶段三：`schema/nodes.py`

### 本阶段文件

```text
main/parse/schema/nodes.py
```

### 本阶段依赖

```text
标准库：abc、copy、dataclasses、pathlib、typing、collections.abc
项目内：context.MISSING、context.ConfigPath、errors 中的 Schema 子异常
第三方库：无
```

### 本阶段实现结果

实现 Schema 类，不实现任务或组件的具体字段。建议所有节点使用不可变 dataclass，让一个 Schema 可以安全地被多个 Spec 复用。

### `SchemaNode`：抽象基类

实现形式：定义一个继承 `abc.ABC` 的类，并使用 `abc.abstractmethod` 声明抽象实例方法。

这里必须先明确 `normalize` 的含义：

> `normalize` 接收“当前 YAML 节点的原始 Python 值”，按照当前 SchemaNode 描述的结构规则，返回一个新的、形态统一的 Python 值。

它不是把任意错误值强行转换成正确值，也不是执行全部业务校验。它只完成结构层规范化：

```text
识别字段是否缺失
处理普通默认值
判断 None 是否允许
检查基础 Python 类型
递归处理 dict/list 子节点
规范化有限字符串或 Registry key
复制容器，避免修改原始 YAML 对象
```

它不负责：

```text
数字业务范围
组件能力兼容
文件是否存在
跨字段关系
构造 Model/Dataset/Optimizer
具体多态 params 的默认值合并
```

#### 抽象方法签名

```text
normalize(
    self,
    value: Any,
    path: ConfigPath,
    context: ParseContext,
) → Any
```

每个参数的含义：

```text
self
→ 当前 SchemaNode 实例
→ 保存该节点的 required、default、fields 等规则

value
→ 当前 YAML 节点对应的原始 Python 值
→ 字段不存在时传入 MISSING

path
→ value 在 YAML 配置树中的逻辑位置
→ 只用于递归生成子路径和报告错误

context
→ Bootstrap 已识别出的 mode、task 和组件选择
→ 普通 Scalar/Mapping 可能暂时不用
→ RegistryKey/Polymorphic 节点会使用
```

返回值：

```text
成功
→ 返回规范化后的当前节点值

可选字段缺失且没有默认值
→ 返回 MISSING

配置不符合当前节点结构规则
→ 抛出 ConfigError 子类
```

#### 抽象基类自身做什么

`SchemaNode` 自身不实现通用 `normalize` 逻辑，因为不同节点处理方式不同：

```text
ScalarSchema
→ 处理单个值、缺失、None、默认值和基础类型

MappingSchema
→ 处理 dict、字段集合以及对子字段递归

SequenceSchema
→ 处理 list 以及对列表元素递归

PathSchema
→ 处理字符串/Path 的基础形态

PolymorphicSchema
→ 处理 type/params 公共外壳
```

因此抽象基类的第一版只定义接口，不需要构造函数，也不应调用 `super(ABC).__init__()`。

概念实现结构：

```text
class SchemaNode(ABC):
    @abstractmethod
    def normalize(self, value, path, context):
        raise NotImplementedError
```

注意这里是：

```text
class SchemaNode(ABC)
```

不是：

```text
def SchemaNode(ABC)
```

同时实例方法第一个参数必须是 `self`。

`merge()` 放在 `schema/merge.py` 的通用函数中；`walk()` 放在 `schema/walk.py`。这样节点类只描述自身结构和自身规范化行为。

#### 一次完整调用示例

原始 YAML：

```yaml
experiment:
  name: demo
```

Root Mapping 在处理 `experiment.seed` 时发现字段不存在，调用逻辑为：

```text
seed_schema.normalize(
    value=MISSING,
    path=config.experiment.seed,
    context=parse_context,
)
```

若 `seed_schema.default == 42`，返回：

```text
42
```

父 Mapping 把结果写入新字典，最终得到：

```text
{"experiment": {"name": "demo", "seed": 42}}
```

原始 YAML 对象仍然保持：

```text
{"experiment": {"name": "demo"}}
```

再例如原始 YAML：

```yaml
runtime:
  device:
    count: "one"
```

若 `count` 的 ScalarSchema 只允许 `int`，调用 `normalize()` 后不是把 `"one"` 转成整数，而是抛出 `TypeMismatchError`。这说明规范化只处理明确、安全的规则，不猜测用户意图。

#### 所有节点必须遵守的不变量

```text
不修改传入 value
容器成功时返回新 dict/list
默认可变值使用 deepcopy
错误路径使用传入的 ConfigPath
不捕获并吞掉 ConfigError
不执行当前节点职责之外的业务逻辑
```

#### 默认值的统一契约

所有声明 `default` 的节点必须遵守同一条规则，不能按节点临时改变：

```text
default
→ Schema 定义者提供的、已经符合当前节点“规范化输出契约”的值

value is MISSING 且 default is not MISSING
→ 返回 deepcopy(default)
→ 流程立即结束
→ 不再次执行当前节点的类型转换、normalizer 或子 Schema.normalize()
```

采用这条规则的原因：

```text
ScalarSchema 当前已经直接复制 default
→ SequenceSchema 不能单独重新规范化 default
→ 将来若 MappingSchema 拥有 default，也必须采用同一规则
```

因此职责分为：

```text
__post_init__()
→ 检查 default 的节点级基本输出类型和当前节点可以无上下文判断的局部约束
→ 例如 Sequence default 必须是 list，长度满足 min/max
→ 例如 Enum default 必须是 canonical value

normalize()
→ 用户实际提供的 value 才经过完整类型检查、转换和子节点递归
→ 缺失字段命中 default 时只 deepcopy 后返回
```

默认值不会获得用户配置的真实 `path` 和 `context`，所以构造阶段不得为了验证默认值而伪造路径或调用依赖上下文的子 `normalize()`。默认容器中的深层元素由 Schema 定义者负责保证符合子 Schema；后续若决定“默认值也完整规范化”，必须一次性修改所有拥有 default 的节点协议，不能只修改某一个类。

`allow_none` 只约束用户或 Override 实际传入的 `value=None`，不自动表示 `default=None` 合法。每个节点必须单独声明自己的 default 输出类型；例如当前 `SequenceSchema.default` 只能是 `list` 或 `MISSING`。

#### 容器节点的递归边界

```text
MappingSchema
→ 配置值是 Mapping
→ fields 中的 child_schema 才是 SchemaNode
→ 调用 child_schema.normalize(raw_child, child_path, context)

SequenceSchema
→ 配置值是 list
→ 列表中的 item 是普通配置值，不是 Schema
→ item_schema 才是 SchemaNode
→ 调用 item_schema.normalize(item, item_path, context)
```

容器节点不得根据运行时值的 `dict/list` 类型自动猜测子 Schema。只有显式声明为 `MappingSchema` 或 `SequenceSchema` 才递归相应结构；`ScalarSchema(python_types=(dict,))` 表示把整个 dict 当作一个叶子值。

#### 逐类实现说明的强制粒度

从本节开始，每个节点的实现说明都必须覆盖以下内容；后续开发与审查不得只写字段列表和几句概念描述：

```text
1. 类字段及每个默认值的准确语义
2. __post_init__() 按真实执行顺序的每一项检查
3. 每项检查使用 TypeError、ValueError 或哪一个 ConfigError 子类，以及原因
4. normalize() 按真实控制流排列的 MISSING、default、None、类型和节点特有分支
5. 每个分支返回什么、是否继续执行、是否修改输入
6. 直接上游传入什么，直接下游调用什么
7. 使用的标准库/项目 API、参数、返回值和容易写错的布尔短路顺序
8. 与相似节点共用的规则
9. 当前节点独有、不能机械照抄的规则
10. 当前完成标准和暂时禁止实现的后续职责
```

说明可以不提供完整代码，但必须细致到开发者能够按控制流逐行实现，并能判断每个异常分支为何存在。

#### 构造错误与配置错误的统一划分

Schema 节点有两类失败，必须按错误来源选择异常，不能笼统写成“报错”：

```text
节点定义错误
→ 开发者在 Python 中使用了非法构造参数
→ 在节点 __post_init__() 中发现
→ 参数类型错误抛 TypeError
→ 参数取值或参数组合错误抛 ValueError

配置值错误
→ Schema 定义本身合法，但 YAML/Override 提供的 value 不符合规则
→ 在 normalize() 中发现
→ 抛 errors.py 中对应的 ConfigError 子类
```

统一约定：

```text
TypeError
→ 构造参数的 Python 类型不符合接口
→ 例如 python_types 不是 tuple、normalizer 不是 callable

ValueError
→ 构造参数类型正确，但值或参数组合非法
→ 例如 python_types 为空、required=True 同时提供 default、min_items < 0

MissingFieldError
→ value is MISSING，且节点 required=True 且没有 default

NullNotAllowedError
→ value is None，但节点 allow_none=False

TypeMismatchError
→ YAML/Override 的具体值不是该节点允许的 Python 类型

UnknownFieldError
→ Mapping 输入包含未声明字段，且 allow_unknown=False

SchemaError
→ 配置规范化期间出现无法归入上述更具体类型的 Schema 结构错误；不要在已有具体异常可用时退化为它

UnknownRegistryKeyError
→ RegistryKeySchema 收到的字符串无法在 Registry 中解析
```

构造错误不使用 `ConfigError`，因为它们不是用户 YAML 的错误，也没有可靠的配置 `path`。配置错误必须携带当前 `path`、`actual_value`、`expected` 和节点来源 `source`（能够确定时），由入口统一格式化。

### `ScalarSchema`：普通叶子节点类

构造字段：

```text
python_types: tuple[type, ...]
required: bool = false
allow_none: bool = false
default: Any = MISSING
normalizer: Callable | None = None
```

构造时检查及失败类型：

```text
python_types 不是 tuple
→ TypeError

python_types 是空 tuple
→ ValueError

python_types 中任一元素不是 type 对象
→ TypeError

required 不是严格 bool
→ TypeError

allow_none 不是严格 bool
→ TypeError

required=True 且 default is not MISSING
→ ValueError
→ 必填字段和默认值语义冲突，二者不能同时启用

normalizer 既不是 None 也不可调用
→ TypeError
```

构造成功不返回额外结果；以上检查建议由 `__post_init__()` 完成。错误消息至少包含参数名、实际值或实际类型和预期规则。这里抛出的 `TypeError`/`ValueError` 表示 Python Schema 定义错误，不是 YAML 配置错误，因此不使用 `ConfigError` 子类。

`normalize()` 顺序及失败类型：

```text
1. value is MISSING
2. 若 default 非 MISSING，返回 deepcopy(default)
3. 否则若 required=True，抛 MissingFieldError
4. 否则返回 MISSING，交给父 Mapping 决定不写入结果
5. value is None 且 allow_none=True，返回 None
6. value is None 且 allow_none=False，抛 NullNotAllowedError
7. value 的类型不属于 python_types，抛 TypeMismatchError
8. 有 normalizer 时调用 normalizer(value)
9. normalizer 正常返回，返回其结果；没有 normalizer 时返回原 value
```

`normalizer` 的异常边界：

```text
normalizer 主动抛 ConfigError 子类
→ 原样向上传递，不捕获、不改写

normalizer 因输入值违反其明确转换规则而抛 ValueError/TypeError
→ 包装为 ParameterNormalizationError，并使用 raise ... from error 保留原异常

normalizer 因实现缺陷抛出其他异常
→ 不伪装成用户配置错误，原样向上传递
```

若 `normalizer` 返回值仍需满足 `python_types`，必须在接口设计中明确选择一种规则。本文第一版约定为：`python_types` 只约束 normalizer 的输入值，normalizer 的输出契约由该 normalizer 自身及后续 Validator 负责，不在 `ScalarSchema` 中进行第二次类型检查。

严格整数要注意 `bool` 是 `int` 子类。需要严格整数时，不应只使用 `isinstance(value, int)`；可以在 normalizer/Validator 中使用 `type(value) is int`。

### `MappingSchema`：固定字典节点类

构造字段及失败类型：

```text
fields 不是 Mapping
→ TypeError

fields 中的字段名不是 str
→ TypeError

fields 中的字段名是空字符串
→ ValueError

fields 中的节点不是 SchemaNode 实例
→ TypeError

allow_unknown 不是严格 bool
→ TypeError
```

构造参数合法性检查建议由 `__post_init__()` 完成。错误消息至少包含参数名、实际值或类型和预期规则。这些是 Schema 定义错误，不是 YAML 配置错误，因此使用内置 `TypeError`/`ValueError`，不使用 `ConfigError` 子类。

`normalize()` 顺序及失败类型：

```text
1. value 不是 Mapping
   → TypeMismatchError
2. 计算 value.keys - fields.keys
3. allow_unknown=False 且存在未知字段
   → UnknownFieldError
4. 创建空结果 dict
5. 按 fields 的稳定顺序读取 value.get(name, MISSING)
6. 调用子 Schema.normalize
7. 子结果不是 MISSING 时写入结果
8. allow_unknown=True 时 deepcopy 未知字段到结果
9. 返回新 dict
```

子节点抛出的 `MissingFieldError`、`NullNotAllowedError`、`TypeMismatchError` 等 `ConfigError` 必须原样向上传递，并保留子节点路径；Mapping 不应将它们统一包装成其他异常。

为什么返回新 dict：默认值、路径和 params 后续还会继续规范化，不能污染原始 YAML 对象。

### `SequenceSchema`：列表节点类

字段：

```text
item_schema: SchemaNode
required: bool = false
default: Any = MISSING
min_items: int = 0
max_items: int | None = None
allow_none: bool = false
```

`required`、`default`、`allow_none` 的缺失语义与 `ScalarSchema` 一致，但 `default` 的类型契约由节点自身决定：

```text
required=True 且 default is not MISSING
→ Schema 定义错误，构造阶段抛 ValueError

value is MISSING 且 default 是合法 list
→ 返回 default 的 deepcopy

value is MISSING 且 required=True 且无 default
→ 抛 MissingFieldError

value is MISSING 且 required=False 且无 default
→ 返回 MISSING

value is None 且 allow_none=True
→ 返回 None

value is None 且 allow_none=False
→ 抛 NullNotAllowedError
```

对 `SequenceSchema` 而言，`default` 的合法类型只有 `list` 或 `MISSING`：

```text
default is not MISSING 且 default 不是 list
→ Schema 定义错误，构造阶段抛 TypeError

包括 default=None
→ 即使 allow_none=True 也仍然抛 TypeError
```

`allow_none` 只约束配置或 Override 实际传入的 `value`，不改变 `default` 的类型契约。

构造字段及失败类型：

```text
item_schema 不是 SchemaNode 实例
→ TypeError

required/allow_none 不是严格 bool
→ TypeError

required=True 且 default is not MISSING
→ ValueError

default is not MISSING 且 default 不是 list
→ TypeError
→ 包括 default=None，即使 allow_none=True 也不例外

default 是 list 且长度小于 min_items
→ ValueError

default 是 list 且 max_items is not None 且长度超过 max_items
→ ValueError

min_items 不是严格 int
→ TypeError

min_items < 0
→ ValueError

max_items 既不是 None 也不是严格 int
→ TypeError

max_items < 0
→ ValueError

max_items is not None 且 max_items < min_items
→ ValueError
```

构造参数合法性检查建议由 `__post_init__()` 完成。这些是 Schema 定义错误，使用内置 `TypeError`/`ValueError`。

`normalize()` 顺序及失败类型：

```text
1. value is MISSING
   ├── default is not MISSING → 返回 deepcopy(default)，流程结束
   ├── required=True → MissingFieldError
   └── required=False → 返回 MISSING
2. value is None
   ├── allow_none=True → 返回 None
   └── allow_none=False → NullNotAllowedError
3. value 不是 list
   → TypeMismatchError
4. len(value) < min_items 或 max_items is not None 且 len(value) > max_items
   → SchemaError
   → 错误消息包含 path、实际长度、min_items 和 max_items
5. result = []
6. enumerate(value)
7. item_schema.normalize(item, path.index(i), context)
8. 子结果不是 MISSING 时 append 到 result
9. 返回新的 result list
```

`default` 是已经符合本 `SequenceSchema` 输出契约的默认列表：构造阶段检查其类型和长度；缺失字段使用默认值时只 `deepcopy(default)` 并直接返回，不再次调用 `item_schema.normalize()`。这与 `ScalarSchema` 默认值直接复制返回的规则一致。默认列表的元素由 Schema 定义者负责保证符合 `item_schema`。

列表元素的 `MISSING` 结果不进入返回列表，`None` 结果则保留。列表元素的具体错误由 `item_schema` 决定，并保留带索引的路径；`SequenceSchema` 不应将元素错误统一改写。

YAML 中 tuple 不作为合法输入，避免 Parser 接受 YAML 无法自然表达的结构。

### `EnumSchema`：有限字符串节点类

#### 职责与特殊性

`EnumSchema` 在概念上是“字符串类型检查 + 有限选项查询 + canonical 输出”。它和 `ScalarSchema(python_types=(str,))` 共用缺失、默认值和 `None` 语义，但有三个独有规则：

```text
choices 可以是 set[str] 或 Mapping[str, str]
case_sensitive 决定输入查询键如何匹配
成功返回 canonical value，不一定返回用户原始字符串
```

它只用于静态有限值，例如 `scheduler.interval`。`run.mode`、`task.type` 和组件 `type` 属于 Registry key，应使用 Bootstrap/RegistryKeySchema，不能维护重复 Enum。

#### 字段

```text
choices: Mapping[str, str] | set[str]
case_sensitive: bool = false
required: bool = false
allow_none: bool = false
default: Any = MISSING
```

两种 `choices` 的输出语义：

```text
set[str]
→ 每个集合元素同时是允许的输入和 canonical 输出

Mapping[str, str]
→ key 是允许的输入或别名
→ value 是返回的 canonical 输出
```

#### `__post_init__()` 的逐步流程

按以下顺序检查，前一步失败后立即抛错，不继续访问依赖其前置条件的数据：

```text
1. 检查 choices 的容器类型
   ├── 不是 Mapping 且不是 set → TypeError
   └── 合法 → 继续

2. 检查 choices 非空
   ├── 为空 → ValueError
   └── 非空 → 继续

3. 若 choices 是 Mapping，按 items() 遍历
   ├── key 不是 str → TypeError
   ├── value 不是 str → TypeError
   ├── key.strip() 为空 → ValueError
   ├── value.strip() 为空 → ValueError
   └── 全部合法 → 继续

4. 若 choices 是 set，逐个遍历元素
   ├── 元素不是 str → TypeError
   ├── 元素 strip 后为空 → ValueError
   └── 全部合法 → 继续

5. 检查 case_sensitive
   ├── type(case_sensitive) is not bool → TypeError
   └── 合法 → 继续

6. 检查 required
   ├── type(required) is not bool → TypeError
   └── 合法 → 继续

7. 检查 allow_none
   ├── type(allow_none) is not bool → TypeError
   └── 合法 → 继续

8. 检查 required/default 组合
   ├── required=True 且 default is not MISSING → ValueError
   └── 合法 → 继续

9. 检查 default 的输出类型
   ├── default is MISSING → 无默认值，构造完成
   ├── default 不是 str，包括 None → TypeError
   └── default 是 str → 继续

10. 检查 default 是 canonical value
    ├── choices 是 set → default 必须等于集合中的某个 canonical 元素
    ├── choices 是 Mapping → default 必须等于 Mapping.values() 中的某个 canonical value
    ├── 不满足 → ValueError
    └── 满足 → 构造完成
```

默认值必须是 canonical 输出，不接受只存在于 Mapping key 中的别名。例如：

```text
choices={"training": "train", "train": "train"}
default="train"
→ 合法

default="training"
→ ValueError；它是输入别名，不是输出 canonical value
```

这遵守统一默认值契约：缺失时直接 `deepcopy(default)` 返回，不再做 choices 查询。

#### 大小写和空白的匹配规则

第一版固定为：

```text
case_sensitive=True
→ 用户 value 原样作为 lookup key
→ 不 strip，不 casefold

case_sensitive=False
→ 用户 value 先 strip()，再 casefold()
→ choices 的输入侧也使用相同规则生成比较键
→ 返回值始终使用 choices 中声明的 canonical 原文
```

构造阶段还必须检查“规范化后的输入键冲突”：

```text
case_sensitive=False 时
→ "TRAIN" 与 " train " 的比较键都为 "train"
→ 如果二者映射到不同 canonical value，抛 ValueError
→ 如果映射到同一 canonical value，可视为重复别名；第一版也建议抛 ValueError，要求 Schema 定义去重
```

set 在 `case_sensitive=False` 时若多个元素规范化为同一比较键，同样抛 ValueError。这个检查是 `EnumSchema` 特有的，不能只验证原始容器没有重复项；Python set 只能消除原字符串重复，不能消除 casefold/strip 后的冲突。

#### `normalize()` 的逐步流程

```text
1. value is MISSING
   ├── default is not MISSING → 返回 deepcopy(default)，流程结束
   ├── required=True → 抛 MissingFieldError
   └── required=False → 返回 MISSING

2. value is None
   ├── allow_none=True → 返回 None，流程结束
   └── allow_none=False → 抛 NullNotAllowedError

3. 检查用户值类型
   ├── value 不是 str → 抛 TypeMismatchError
   └── value 是 str → 继续

4. 生成 lookup key
   ├── case_sensitive=True → lookup_key=value
   └── case_sensitive=False → lookup_key=value.strip().casefold()

5. 查询 choices
   ├── choices 是 Mapping
   │   → 对输入侧 key 使用相同匹配规则
   │   → 找到时返回该 key 对应的 canonical value
   └── choices 是 set
       → 对集合元素使用相同匹配规则
       → 找到时返回集合中声明的 canonical 原文

6. 未找到 lookup key
   → 抛 SchemaError
   → 消息包含 path、实际 value 和允许的输入选项
```

`EnumSchema` 不使用 `normalizer`，因此没有 `ParameterNormalizationError` 分支。它不修改 `choices` 或用户输入，不返回 lookup key 的临时字符串；成功时返回 Schema 中声明的 canonical value。

#### 直接数据流

```text
父 MappingSchema/SequenceSchema
→ 传入 value、ConfigPath、ParseContext
→ EnumSchema 处理缺失/None/字符串/选项
→ 返回 MISSING、None 或 canonical str
→ 父容器将结果写入新 dict/list
```

`context` 第一版不使用，但必须保留统一 `SchemaNode.normalize()` 签名。

#### 容易写错的地方

```text
choices 元素类型错误应抛 TypeError，不是 ValueError
默认值检查的是 canonical 输出集合，不是输入别名集合
case_sensitive=False 时 choices 输入侧也必须 strip/casefold，不能只处理用户 value
返回 canonical 原文，不能直接返回 lookup_key
不要让 default 再次经过 choices 查询；它在构造阶段已经验证
不要把未知枚举值报成 TypeMismatchError；它的 Python 类型已经是 str，应报 SchemaError
```

#### 当前完成标准

```text
所有构造参数按上述顺序检查
规范化输入键冲突可诊断
MISSING/default/required 行为与统一协议一致
value=None 只由 allow_none 控制
非 str 用户值抛 TypeMismatchError
set 和 Mapping 都能返回 canonical 原文
未知字符串抛 SchemaError
不修改输入和 choices
```

当前步骤只实现 `EnumSchema`；不要同时开始 `RegistryKeySchema`、`PathSchema` 或 PolymorphicSchema。

### `RegistryKeySchema`：Registry 键节点类

#### 职责与特殊性

`RegistryKeySchema` 处理的仍是字符串配置值，但允许集合不保存在节点内，而由 Registry 动态提供。它与 `EnumSchema` 的区别必须保持：

```text
EnumSchema
→ 静态 choices
→ 适合 interval 等固定有限值

RegistryKeySchema
→ 调用 registry.resolve_key(raw_key)
→ 支持注册、别名和 canonical key
→ 适合 task、component type 等扩展点
```

#### 字段

```text
registry
required: bool = true
allow_none: bool = false
```

第一版不提供 `default`：Registry key 通常是判别字段，缺失时应明确失败；若某调用点设置 `required=False`，缺失时返回 `MISSING`。

#### `__post_init__()` 的逐步流程

```text
1. 检查 registry 接口
   ├── registry is None → TypeError
   ├── 没有可调用的 resolve_key 属性 → TypeError
   └── 合法 → 继续

2. 检查 required
   ├── type(required) is not bool → TypeError
   └── 合法 → 继续

3. 检查 allow_none
   ├── type(allow_none) is not bool → TypeError
   └── 合法 → 构造完成
```

当前节点层不应为了类型检查导入未来具体 `BaseRegistry` 而制造循环依赖；第一版按最小协议检查 `resolve_key` 可调用。Registry 的完整定义合法性由 Registry 自身负责。

#### `normalize()` 的逐步流程

```text
1. value is MISSING
   ├── required=True → MissingFieldError
   └── required=False → 返回 MISSING

2. value is None
   ├── allow_none=True → 返回 None
   └── allow_none=False → NullNotAllowedError

3. value 不是 str
   → TypeMismatchError

4. value.strip() 为空
   → SchemaError
   → 消息包含 path 和“Registry key 不能为空”

5. 调用 registry.resolve_key(value)
   ├── 成功 → 返回 canonical key
   ├── UnknownRegistryKeyError → 原样向上传递
   └── Registry 抛出的其他异常 → 原样传播，不在节点中猜测改写
```

节点不自行执行 `strip/casefold` 后查 Registry；它只在调用前拒绝空字符串。别名、大小写规则和 canonical 输出必须由 Registry 的唯一实现负责，避免 Bootstrap 与 Schema 使用不同规则。

#### 直接数据流与完成标准

```text
父容器传入 raw key
→ RegistryKeySchema 检查缺失/None/str/空白
→ registry.resolve_key(raw key)
→ 返回 canonical key
→ 父容器写入新配置
```

完成标准：可选缺失返回 `MISSING`，空字符串不是未知键，未注册键保持 `UnknownRegistryKeyError`，成功值只来自 Registry，不修改输入。当前节点不注册 Spec、不创建组件、不访问 Bootstrap selections。

### `PolymorphicSchema`：多态组件外壳类

#### 职责与特殊性

它只规范化组件节点的公共外壳：

```text
{discriminator, params_field, 其他公共字段}
```

它不能读取具体 `params_schema`，因为具体参数 Schema 取决于 Bootstrap 已选择的 ComponentSpec。第一阶段保留 `params=None` 或用户 Mapping；第二阶段 `normalize_polymorphic_nodes()` 才合并默认参数并调用具体 Schema。

#### 字段

```text
category: str
discriminator: str = "type"
params_field: str = "params"
common_schema: MappingSchema
```

`common_schema` 必须显式包含 discriminator 与 params_field 对应的子 Schema；前者通常是 `RegistryKeySchema`，后者负责只接受 `None` 或 Mapping 的公共形态。

#### `__post_init__()` 的逐步流程

```text
1. 检查 category
   ├── 不是 str → TypeError
   ├── strip 后为空 → ValueError
   └── 合法 → 继续

2. 检查 discriminator
   ├── 不是 str → TypeError
   ├── strip 后为空 → ValueError
   └── 合法 → 继续

3. 检查 params_field
   ├── 不是 str → TypeError
   ├── strip 后为空 → ValueError
   └── 合法 → 继续

4. discriminator == params_field
   → ValueError；两个角色不能指向同一字段

5. common_schema 不是 MappingSchema
   → TypeError

6. discriminator 不在 common_schema.fields
   → ValueError

7. params_field 不在 common_schema.fields
   → ValueError

8. 全部通过
   → 构造完成
```

#### `normalize()` 的逐步流程

```text
1. value 不是 Mapping
   → TypeMismatchError

2. 调用 common_schema.normalize(value, path, context)
   → 返回新的公共结构 dict
   → MissingFieldError、UnknownFieldError、TypeMismatchError、UnknownRegistryKeyError 原样传播

3. 从规范化结果读取 discriminator
   ├── 不存在 → SchemaError；说明 common_schema 定义不符合 PolymorphicSchema 契约
   └── 存在 → 已由其子 Schema 规范化为 canonical key

4. 从规范化结果读取 params_field
   ├── 不存在 → SchemaError；说明 common_schema 定义不符合契约
   ├── 值是 None → 保留 None
   ├── 值是 Mapping → 确保结果是新的 dict/Mapping 结果
   └── 其他类型 → TypeMismatchError（通常应已由 common_schema 拒绝）

5. 返回 common_schema 产生的新 dict
```

此节点不再次查询 Registry、不调用 `context.get_selection()`、不合并 `default_params`、不运行组件 Validator。那些职责属于第二阶段 `normalize_polymorphic_nodes()`。

#### 直接数据流与完成标准

```text
原始组件 Mapping
→ common_schema.normalize
→ type canonicalized、params 保持 None/Mapping
→ 返回公共外壳
→ 后续按 CompiledSchema.polymorphic_nodes 定位
→ selection.spec.normalize_params(...)
```

完成标准：输入不修改；公共字段错误保留真实子路径；`params` 尚未保证为最终 dict；节点内没有具体任务、组件名称或参数分支。

### `PathSchema`：路径描述节点类

#### 职责与特殊性

`PathSchema` 只把结构层路径输入规范化为 `pathlib.Path`，不负责得到最终绝对路径，也不访问文件系统：

```text
本节点 normalize
→ str/pathlib.Path 转 pathlib.Path（可以仍是相对路径）

resolve_paths
→ 按 base 和依赖图得到绝对路径

Validator
→ 检查 exists、is_file/is_dir、后缀和目录非空
```

这三个阶段不得合并。建议以组合或继承 `ScalarSchema` 的方式复用 MISSING/default/required/None 语义，但最终公共字段必须明确包含：

```text
required: bool = false
default: pathlib.Path | MISSING = MISSING
allow_none: bool = false
kind: "file" | "directory"
base: PathBase
allowed_suffixes: tuple[str, ...] = ()
must_exist: bool = false
non_empty: bool = false
resolver_name: str | None = None
```

默认值遵守统一契约：必须已经是 `pathlib.Path` 输出值；字符串只能作为用户输入，不能作为 default，缺失命中默认值时直接 deepcopy 返回。

#### `__post_init__()` 的逐步流程

```text
1. 检查 required/allow_none/must_exist/non_empty 为严格 bool
   → 任一错误抛 TypeError

2. required=True 且 default is not MISSING
   → ValueError

3. default is not MISSING 且不是 pathlib.Path
   → TypeError；包括 default=None 和 default=str

4. kind 不是 str
   → TypeError

5. kind 不属于 {"file", "directory"}
   → ValueError

6. base 类型不符合 PathBase 协议
   → TypeError

7. base 值非法或字段引用为空
   → ValueError

8. allowed_suffixes 不是 tuple
   → TypeError

9. 遍历 allowed_suffixes
   ├── 元素不是 str → TypeError
   ├── 元素 strip 后为空 → ValueError
   └── 建议要求以 "." 开头，否则 ValueError

10. resolver_name 既不是 None 也不是 str
    → TypeError

11. resolver_name 是 str 但 strip 后为空
    → ValueError

12. non_empty=True 且 kind!="directory"
    → ValueError；非空语义只适用于目录

13. 全部通过
    → 构造完成
```

`must_exist=False` 时仍可声明 `kind` 和后缀；这些信息供后续创建目标路径或其他 Validator 使用，不在 normalize 中检查。

#### `normalize()` 的逐步流程

```text
1. value is MISSING
   ├── default is not MISSING → 返回 deepcopy(default)，流程结束
   ├── required=True → MissingFieldError
   └── required=False → 返回 MISSING

2. value is None
   ├── allow_none=True → 返回 None
   └── allow_none=False → NullNotAllowedError

3. value 是 pathlib.Path
   → 返回一个语义等价的 pathlib.Path
   → 不 resolve，不检查存在性

4. value 是 str
   ├── value.strip() 为空 → SchemaError
   ├── 调用 pathlib.Path(value)
   ├── 转换成功 → 返回 Path
   └── Path 构造抛 TypeError/ValueError/OSError
       → 包装为 TypeMismatchError，并 raise ... from error

5. 其他类型
   → TypeMismatchError
```

不要在这里调用：

```text
Path.resolve()
Path.exists()
Path.is_file()
Path.is_dir()
Path.iterdir()
```

#### 直接数据流与完成标准

```text
父容器传入原始路径值
→ PathSchema 返回 MISSING/None/pathlib.Path
→ CompiledSchema 收集该 PathSchema 的 ConfigPath
→ resolve_paths 按拓扑顺序处理
→ Validator 检查文件系统语义
```

完成标准：相对路径仍保持相对；不读取磁盘；默认值只接受 Path；路径转换异常保留 cause；`kind/base/resolver_name` 只保存元数据，不在 normalize 中越权执行解析或推导。

### 本阶段不实现

```text
Schema 合并
Schema 遍历收集
具体 QA/MLM Schema
文件存在性
业务数值范围
params 默认值合并
```

### 本阶段完成标准

```text
每个节点都有明确构造参数
normalize 不修改输入
默认可变值使用 deepcopy
列表错误路径带索引
Mapping 拒绝未知字段
PathSchema 不访问磁盘
所有节点可用小型人工 Schema 独立测试
```

## 30.6 阶段四：Schema 遍历、合并与编译

### 本阶段文件

```text
main/parse/schema/walk.py
main/parse/schema/merge.py
main/parse/schema/compile.py
```

### 本阶段依赖

```text
标准库：copy、dataclasses、typing
项目内：schema.nodes、context.ConfigPath、errors.SchemaConflictError
第三方库：无
```

### `walk.py`：实现纯函数遍历器

数据类 `SchemaEntry`：

```text
path: ConfigPath
node: SchemaNode
```

函数：

```text
walk_schema(root, root_path=ConfigPath.root("config"))
→ Iterator[SchemaEntry]
```

流程：

```text
先 yield 当前节点
→ Mapping 遍历 fields
→ Sequence 遍历 item_schema，使用特殊 item 标记而不假装真实索引
→ Polymorphic 遍历 common_schema
```

该函数不规范化配置，只遍历 Schema 对象。

### `merge.py`：实现 Schema 合并纯函数

主接口：

```text
merge_schema(left, right, path, left_source, right_source) → SchemaNode
```

实现方式：使用节点类型分派的普通函数，不需要创建 `SchemaMerger` 大类。

流程：

```text
1. 节点具体类型不同 → SchemaConflictError
2. Mapping → 合并同名字段，独有字段直接保留
3. Sequence → 递归合并 item_schema，检查 min/max
4. Scalar → 检查类型、required、allow_none、default、normalizer
5. Path → 额外检查 kind/base/suffix/resolver
6. Polymorphic → 检查 category/discriminator/params_field，再合并 common_schema
7. 任一同名节点的约束不兼容
   → SchemaConflictError
   → 错误消息包含 path、left_source、right_source、冲突字段和两侧值
8. 返回新的 SchemaNode
```

Schema 合并阶段不会接收用户 YAML 值；因此这里的失败属于 Schema 定义冲突，应使用 `SchemaConflictError`，而不是 `TypeMismatchError` 或 `ConfigValidationError`。

禁止修改 left/right。

默认值比较注意：若是复杂可变对象，不要求支持任意对象深度比较；第一版只允许可稳定比较的 YAML 值作为默认值。

### `compile.py`：实现编译结果和编译函数

数据类 `SpecBundle`：

```text
root_schema
mode_spec
task_spec
component_specs
```

数据类 `CompiledSchema`：

```text
root: SchemaNode
polymorphic_nodes: tuple[SchemaEntry, ...]
path_nodes: tuple[SchemaEntry, ...]
validator_refs: tuple[Any, ...]
schema_sources: Mapping[ConfigPath, tuple[str, ...]]
```

主函数：

```text
compile_schema(bundle: SpecBundle) → CompiledSchema
```

流程：

```text
1. deepcopy root_schema
2. 合并 mode_spec.schema_contribution
3. 合并 task_spec.schema_contribution
4. 按 ComponentSelection 稳定顺序合并组件 contribution
5. walk_schema 最终 root
6. 收集 PolymorphicSchema
7. 收集 PathSchema
8. 收集 Validator 引用
9. 生成 schema_sources
10. 返回 CompiledSchema
```

为什么返回 dataclass：后续 Normalize、Path 和 Validator 都需要相同的已编译元数据，不应各自重新遍历和组合。

### 本阶段完成标准

```text
合并不修改输入节点
冲突错误包含配置路径和两个 source
CompiledSchema 元数据使用不可变 tuple
人工 Root+Mode+Task Schema 可成功合并
节点类型、默认值、Path 基准冲突均能报错
```

## 30.7 阶段五：`specs/` 与 `registries/`

### 本阶段文件

```text
main/parse/specs/component.py
main/parse/specs/mode.py
main/parse/specs/task.py
main/parse/specs/validator.py
main/parse/registries/base.py
main/parse/registries/components.py
main/parse/registries/modes.py
main/parse/registries/tasks.py
```

### 本阶段依赖

```text
标准库：dataclasses、typing、collections.abc
项目内：SchemaNode、ConfigPath、ConfigError 子类
第三方库：无
```

### `ValidatorSpec`：数据类

文件：`specs/validator.py`。

构造参数及失败类型：

```text
name、phase、source 不是 str
→ TypeError

name、phase、source 为空字符串
→ ValueError

priority 不是严格 int
→ TypeError

validator 不是 callable
→ TypeError

applies_to 既不是 None 也不可调用
→ TypeError
```

这些属于 ValidatorSpec 定义错误，应在 `__post_init__()` 中发现；不使用 `ConfigValidationError`，因为此时还没有用户配置值。

字段：

```text
name: str
phase: str
priority: int
validator: Callable
applies_to: Callable | None
source: str
```

方法：

```text
should_run(context, config) → bool
run(context, config) → None
```

`run()` 只调用 validator；不要捕获并吞掉 ConfigError。

### `ComponentSpec`：不可变数据类

文件：`specs/component.py`。

字段：

```text
category: str
key: str
aliases: tuple[str, ...]
params_schema: MappingSchema
default_params: Mapping[str, Any]
capabilities: frozenset[str]
normalizers: tuple[Callable, ...]
validators: tuple[ValidatorSpec, ...]
factory: Callable | None
schema_contribution: SchemaNode | None
```

方法及返回：

```text
validate_definition() → None
normalize_params(raw_params, path, context) → dict
validate_params(params, path, context) → None
build(params, build_context) → Any
```

具体行为：

```text
validate_definition
→ key/category 不是字符串或为空
   → TypeError 或 ValueError
→ aliases 不是合法的字符串序列，或包含空字符串
   → TypeError 或 ValueError
→ aliases 规范化后不重复，且不与 canonical key 冲突
   → 冲突时抛 ValueError
→ default_params 不是 Mapping
   → TypeError
→ default_params 不能通过 params_schema
   → SchemaError 或 ParameterNormalizationError
→ factory 既不是 None 也不可调用
   → TypeError

normalize_params
→ raw_params 既不是 None 也不是 Mapping
   → TypeMismatchError
→ deepcopy(default_params)
→ raw_params=None 时不覆盖
→ raw_params 为 Mapping 时覆盖
→ params_schema.normalize
   → 参数结构不合法时抛对应 SchemaError 子类
→ 依次运行 normalizers
   → 参数规范化失败时抛 ParameterNormalizationError
→ 返回 dict

validate_params
→ 组件 Validator 返回 None
→ 校验失败时抛 ConfigValidationError 或其子类

build
→ factory is None
   → ParameterNormalizationError 或专门的构造错误
→ factory 调用失败
   → 原样传播，不能伪装成 YAML 结构错误
→ 否则返回 factory(params=params, context=build_context) 的结果
```

Parser 只调用前 3 个接口；Builder 才调用 `build()`。

### `TaskSpec`：不可变数据类

文件：`specs/task.py`。

字段：

```text
key
aliases
supported_modes: frozenset[str]
task_params_schema
schema_contribution
bootstrap_descriptors
required_capabilities_by_mode
normalizers
validators
cross_validators
builders_by_mode: Mapping[str, Callable]
```

方法：

```text
validate_definition() → None
supports_mode(mode_key) → bool
get_builder(mode_key) → Callable
```

`validate_definition()` 必须检查：

```text
supported_modes 与 builders_by_mode.keys 一致
每个 mode 的能力要求结构合法
每个 Builder 可调用
```

`get_builder()` 不拼接字符串，直接按 mode_key 查映射；不支持时抛 `ModeTaskMismatchError`。

### `ModeSpec`：不可变数据类

文件：`specs/mode.py`。

字段：

```text
key
aliases
schema_contribution
bootstrap_descriptors
required_sections
forbidden_sections
normalizers
validators
runner
```

方法：

```text
validate_definition() → None
get_runner() → Callable
```

`runner` 与 Task Builder 不同：Runner 负责 Train/Test/Inference 的运行过程；Task Builder 负责构造该任务所需组件。`runner` 不是可调用对象时属于 ModeSpec 定义错误，抛 `TypeError`；`get_runner()` 成功返回 callable。

### `BaseRegistry`：有状态普通类

文件：`registries/base.py`。

内部字段：

```text
_name
_spec_type
_specs: dict[str, Spec]
_aliases: dict[str, str]
```

方法：

```text
normalize_key(raw_key) → str
register(spec) → None
resolve_key(raw_key) → str
get(key) → Spec
contains(key) → bool
all_keys() → tuple[str, ...]
```

`register()` 顺序及失败类型：

```text
检查 spec 类型
→ 不是预期 Spec 类型时抛 TypeError

→ spec.validate_definition()
→ Spec 定义字段非法时保留其 TypeError/ValueError

→ 规范化 canonical key
→ key 不是非空字符串时抛 TypeError 或 ValueError

→ 检查 key 重复
→ 已注册同名 key 时抛 ValueError

→ 逐个规范化 alias
→ alias 类型错误或为空时抛 TypeError 或 ValueError

→ 检查 alias 与 key/alias 冲突
→ 冲突时抛 ValueError

→ 写入 _specs/_aliases
→ 返回 None
```

`resolve_key()` 返回 canonical key；`get()` 可以先调用 `resolve_key()`，因此别名和规范键都可查询。

### 分类 Registry

```text
ComponentRegistry
→ 需要 category
→ 每个 category 独立 BaseRegistry

ModeRegistry
→ BaseRegistry[ModeSpec]

TaskRegistry
→ BaseRegistry[TaskSpec]
```

推荐提供模块级 Registry 实例，但不要在导入 `base.py` 时自动导入全部内置任务，避免循环导入。统一注册动作放在 `builtins/register.py`。

### 本阶段完成标准

```text
Spec 都是描述对象，不创建运行组件
Registry 返回 canonical key
别名冲突可诊断
TaskSpec 不拼接 mode/task
ComponentSpec 默认 params 可自校验
Registry 与具体 MLM/QA 类无耦合
```

## 30.8 阶段六：`bootstrap.py`

### 本阶段文件

```text
main/parse/bootstrap.py
```

### 本阶段依赖

```text
标准库：dataclasses、pathlib、typing、collections.abc
项目内：context、registries、specs、errors
第三方库：无
```

### `BootstrapDescriptor`：不可变数据类

字段：

```text
path_pattern: tuple[str | "*", ...]
category: str
required: bool
```

示例：

```text
("components", "model") category="model"
("components", "losses", "*") category="loss"
("evaluation", "metrics", "*", "*") category="metric"
```

`*` 表示遍历 Mapping 值或 Sequence 元素。第一版只支持明确的两种通配场景，不实现通用 XPath。

### 辅助函数

```text
read_required_mapping(raw, key, path) → Mapping
read_required_string(raw, key, path) → str
scan_descriptor(raw_config, descriptor, root_path) → list[(ConfigPath, Mapping)]
```

辅助函数只做 Bootstrap 所需的最小类型检查，并按失败原因抛 BootstrapError 子类：输入节点缺失抛 `BootstrapError` 或 `MissingFieldError`，节点不是 Mapping/字符串类型抛 `TypeMismatchError`，未知 Registry 键抛 `UnknownRegistryKeyError`，Mode 与 Task 不支持组合抛 `ModeTaskMismatchError`。同一阶段的错误不要统一包装成普通 `ValueError`。

### 主函数

```text
bootstrap_config(
    raw_config: Mapping,
    config_path: str | pathlib.Path,
    mode_registry: ModeRegistry,
    task_registry: TaskRegistry,
    component_registry: ComponentRegistry,
    expected_mode: str | None = None,
    validate_runtime: bool = true,
) → ParseContext
```

不建议从模块全局隐式取得 Registry；显式参数更容易测试。`api.py` 可以传入默认 Registry 集合。

执行顺序：

```text
1. 将 config_path 转为绝对 Path，得到 config_dir
2. 检查 raw_config 是 Mapping
3. 读取 run Mapping
4. 读取 run.mode 字符串
5. mode_registry.resolve_key → mode_key
6. expected_mode 非 None 时先规范化，再检查一致
7. 读取 task Mapping 和 task.type
8. task_registry.resolve_key → task_key
9. task_spec.supports_mode(mode_key)
10. 合并 ModeSpec 与 TaskSpec 的 BootstrapDescriptor
11. 按 descriptor 扫描动态节点
12. 读取每个节点的 type
13. component_registry.resolve(category, type)
14. 创建 ComponentSelection
15. 检查同一路径没有重复选择
16. 创建并返回 ParseContext
```

返回对象：只返回 `ParseContext`，不返回修改后的配置。

不做：

```text
不补默认值
不校验 params
不解析 PathSchema
不检查文件存在
不检查能力兼容
不调用 Builder
```

为什么需要单独阶段：完整 Schema 依赖 `type`，而 `type` 又在尚未规范化的 YAML 中。Bootstrap 只解决这个“先知道类型才能选择 Schema”的问题。

### 本阶段完成标准

```text
输入 raw_config 不被修改
mode/task 返回 canonical key
expected_mode 不一致能报错
动态列表路径包含真实索引
未知组件错误包含 category、raw type 和路径
返回 ParseContext.selections 完整且稳定
```

## 30.9 阶段七：`schema/normalize.py`

### 本阶段文件

```text
main/parse/schema/normalize.py
```

### 本阶段依赖

```text
标准库：copy、typing、collections.abc
项目内：CompiledSchema、SchemaNode、ParseContext、ConfigPath、errors
第三方库：无
```

### 路径读写辅助函数

由于多态节点需要按 ConfigPath 读取和替换嵌套值，本文件先实现：

```text
get_value(config, path) → Any
set_value_copy(config, path, value) → dict
```

`set_value_copy()` 返回新根 dict。第一版可以 `deepcopy(config)` 后定位写入，先保证正确性；性能有证据后再改结构共享。

### `normalize_structure()`

签名：

```text
normalize_structure(
    raw_config: Mapping,
    compiled_schema: CompiledSchema,
    context: ParseContext,
) → dict
```

实现方式：直接调用：

```text
compiled_schema.root.normalize(raw_config, ConfigPath.root("config"), context)
```

返回值必须是 dict。若根 Schema 返回其他类型，应抛 SchemaError，表示 Root Schema 定义错误。

本函数不需要先 deepcopy 整棵树，因为每个 SchemaNode 已承诺返回新容器；额外全量 deepcopy 只会重复复制。

### `normalize_polymorphic_nodes()`

签名：

```text
normalize_polymorphic_nodes(
    config: Mapping,
    compiled_schema: CompiledSchema,
    context: ParseContext,
) → dict
```

流程：

```text
1. 创建结果配置副本
2. 按路径稳定排序 polymorphic_nodes
3. get_value 取得节点 dict
4. context.get_selection(node_path)
5. 确认节点 type 等于 selection.key
6. 读取 params
7. selection.spec.normalize_params(params, params_path, context)
8. 确认返回值是 dict
9. 替换节点的 params
10. 返回结果 dict
```

`ComponentSpec.normalize_params()` 内部负责 defaults、用户覆盖、params_schema 和 normalizers，本函数不要重复实现这些组件细节。

结束不变量：

```text
每个 Polymorphic 节点的 type 是 canonical key
每个 params 是 dict
原始和结构规范化配置均不被修改
```

### 本阶段完成标准

```text
Root Schema 规范化可独立运行
默认值只复制一次且不共享
params=None/空 dict/部分 dict 行为正确
Selection 与 type 不一致时报内部配置错误
最终结果可以直接交给 Override/Path 阶段
```

## 30.10 阶段八：`overrides.py`

### 本阶段文件

```text
main/parse/overrides.py
```

### 本阶段依赖

```text
标准库：dataclasses、copy、typing
项目内：ConfigPath、CompiledSchema、ParseContext、SchemaError
第三方库：PyYAML（仅用于把 CLI 字符串安全解析成 Python 标量/列表/dict）
```

### `ConfigOverride`：不可变数据类

字段：

```text
path: ConfigPath
raw_value: Any
source: str
```

### `parse_override()`

签名：

```text
parse_override(raw_text: str, source: str = "cli") → ConfigOverride
```

约定输入格式：

```text
path.to.field=value
```

流程：

```text
只按第一个 = 分割
→ 左侧解析为 ConfigPath
→ 右侧使用 yaml.safe_load
→ 创建 ConfigOverride
```

第一版路径只支持点号字段，不支持 CLI 覆盖列表索引；需要时再明确加入 `[n]` 解析。

### `apply_overrides()`

签名：

```text
apply_overrides(
    config: Mapping,
    overrides: Sequence[ConfigOverride],
    compiled_schema: CompiledSchema,
    context: ParseContext,
) → dict
```

流程：

```text
1. 复制 config
2. 对每个 override 查找对应 SchemaEntry
3. 路径不存在 → UnknownFieldError
4. 检查是否为禁止覆盖的 discriminator
5. 使用目标 node.normalize(raw_value, path, context)
6. 写入结果配置
7. 返回新 dict
```

禁止覆盖：

```text
run.mode
task.type
所有 PolymorphicSchema.type
```

原因：这些字段会改变 Bootstrap 选择和 CompiledSchema。第一版直接拒绝比局部重跑 Bootstrap 更可靠。

`--resume` 不要求用户写通用文本 Override；入口可直接构造：

```text
ConfigOverride(ConfigPath(...run.resume_from), resume_path, "cli:--resume")
```

### 本阶段完成标准

```text
Override 不直接绕过 Schema
输入 config 不变
字符串 true/数字/list 可由 safe_load 转换
判别字段不可覆盖
错误包含 Override source
```

## 30.11 阶段九：路径模块

### 本阶段文件

```text
main/parse/paths/graph.py
main/parse/paths/resolver.py
main/parse/paths/derivers.py
```

### 本阶段依赖

```text
标准库：dataclasses、pathlib、graphlib、typing
项目内：ConfigPath、CompiledSchema、PathSchema、PathResolutionError
第三方库：无
```

### `PathNode`：不可变数据类

建议放在 `paths/graph.py`：

```text
path: ConfigPath
schema: PathSchema
dependency: ConfigPath | None
```

### `collect_path_nodes()`

签名：

```text
collect_path_nodes(compiled_schema: CompiledSchema) → tuple[PathNode, ...]
```

从 `compiled_schema.path_nodes` 转换为路径运行元数据，不重新 walk 整个 Schema。

### `build_dependency_order()`

签名：

```text
build_dependency_order(nodes: Sequence[PathNode]) → tuple[ConfigPath, ...]
```

流程：

```text
PathSchema.base=config_dir → 无依赖
PathSchema.base=字段引用 → 依赖对应 ConfigPath
→ 检查依赖节点存在
→ graphlib.TopologicalSorter
→ static_order()
→ 返回稳定顺序
```

`TopologicalSorter` 抛出的循环异常转换为 `PathDependencyError`，并保留环中路径。

### `derivers.py`

实现命名明确的小函数，不实现一个万能推导器。例如：

```text
derive_train_config(checkpoint_path: pathlib.Path, config, context) → pathlib.Path
```

Deriver 必须是纯路径推导，不读取 YAML、不验证训练配置内容。

### `resolve_paths()`

签名：

```text
resolve_paths(
    config: Mapping,
    compiled_schema: CompiledSchema,
    context: ParseContext,
    deriver_registry: Mapping[str, Callable],
) → dict
```

单节点顺序：

```text
1. 读取当前值
2. None 且有 deriver → 调用 deriver
3. None 且允许 → 保留
4. 字符串转 Path
5. 绝对路径 → resolve(strict=False)
6. 相对路径 → 获取 config_dir 或已解析字段基准
7. base / value 后 resolve(strict=False)
8. 写回结果
```

这里不调用 `is_file()`/`is_dir()`，因为不存在路径仍然需要先得到最终绝对位置，之后 Validator 才能给出统一语义错误。

### 本阶段完成标准

```text
使用 pathlib，不拼接路径字符串
路径依赖按拓扑顺序解析
循环和缺失依赖可诊断
输入配置不变
最终非 None 路径均为绝对 Path
Deriver 不加载其他配置
```

## 30.12 阶段十：Validator Pipeline

### 本阶段文件

```text
main/parse/validation/pipeline.py
main/parse/validation/common.py
main/parse/validation/runtime.py
```

### 本阶段依赖

```text
标准库：math、pathlib、typing
项目内：ValidatorSpec、CompiledSchema、ParseContext、ConfigValidationError
第三方库：torch 只允许在 runtime.py 导入
```

### `common.py`

实现可复用的小校验函数，不创建类：

```text
require_non_empty_string(value, path) → None
require_strict_int(value, path, minimum=None, maximum=None) → None
require_finite_number(value, path, minimum=None, maximum=None) → None
require_file(path_value, path, suffixes=()) → None
require_directory(path_value, path, non_empty=False) → None
```

这些函数成功返回 `None`，失败按校验来源抛出明确异常：

```text
传入值的 Python 类型或结构不符合函数契约
→ TypeMismatchError

业务值、数值范围、文件存在性、后缀或目录非空条件不满足
→ ConfigValidationError

当前设备或 CUDA 运行环境不满足
→ RuntimeValidationError（仅 runtime.py）
```

它们不读取任意配置树，只校验传入值。

### `runtime.py`

实现环境相关 Validator，例如：

```text
validate_device_runtime(context, config) → None
```

只有这里允许导入 `torch`。当 `context.validate_runtime=False` 时，对应 Validator 不运行。

### `pipeline.py`

函数：

```text
collect_validators(
    compiled_schema: CompiledSchema,
    context: ParseContext,
) → tuple[ValidatorSpec, ...]
```

收集来源：

```text
CompiledSchema.validator_refs
ComponentSelection.spec.validators
ModeSpec.validators
TaskSpec.validators/cross_validators
```

如果 ParseContext 当前没有保存 mode_spec/task_spec，可通过一个 `SpecBundle` 或在 context 中增加只读引用；不要在 Pipeline 中重新根据字符串导入具体模块。

排序键：

```text
phase_order[phase], priority, name, source
```

主函数：

```text
validate_config(
    config: Mapping,
    context: ParseContext,
    compiled_schema: CompiledSchema,
) → None
```

流程：

```text
validators = collect_validators(...)
→ 对每个 validator 调用 should_run
→ true 时 run
→ 全部完成后返回 None
```

不捕获 `ConfigError`。如果需要给未知异常增加 source，只能 `raise ... from original`，不能吞掉原始异常。

### 本阶段完成标准

```text
Validator 不修改配置
成功统一返回 None
失败统一 raise
torch 不出现在普通 Schema/Validator 模块
执行顺序稳定
validate_runtime 可以跳过环境检查
```

## 30.13 阶段十一：`pipeline.py` 与 `api.py`

### 本阶段文件

```text
main/parse/pipeline.py
main/parse/api.py
main/parse/snapshot.py
```

### 本阶段依赖

```text
标准库：dataclasses、pathlib、hashlib、json、typing
第三方库：PyYAML
项目内：前面所有通用模块、Registry 集合
```

### `pipeline.py`

数据类 `ParserServices`：

```text
mode_registry
task_registry
component_registry
deriver_registry
root_schema
```

显式集中依赖，避免 `load_config()` 到处读取模块级全局变量。

数据类 `ParsedConfig`：

```text
config: dict
context: ParseContext
compiled_schema: CompiledSchema
schema_fingerprint: str
```

辅助函数：

```text
load_raw_yaml(path: pathlib.Path) → dict
collect_specs(context, services) → SpecBundle
fingerprint_schema(compiled_schema) → str
```

`load_raw_yaml()`：

```text
检查文件存在且是文件
→ 以 UTF-8 读取
→ yaml.safe_load
→ 空文件报 YamlLoadError
→ 顶层非 dict 报 YamlLoadError
→ 返回 dict
```

主函数：

```text
run_parse_pipeline(
    yaml_path,
    services,
    expected_mode=None,
    overrides=(),
    validate_runtime=True,
) → ParsedConfig
```

固定执行顺序：

```text
绝对化 yaml_path
→ load_raw_yaml
→ bootstrap_config
→ collect_specs
→ compile_schema
→ normalize_structure
→ normalize_polymorphic_nodes
→ apply_overrides
→ resolve_paths
→ validate_config
→ fingerprint_schema
→ ParsedConfig
```

任何阶段失败直接向上抛 `ConfigError`；Pipeline 不打印、不退出进程。

### `api.py`

公共函数：

```text
load_config(
    yaml_path,
    *,
    expected_mode=None,
    overrides=None,
    validate_runtime=True,
    services=None,
) → ParsedConfig
```

行为：

```text
services is None
→ 取得 builtins 创建的默认 ParserServices

然后调用 run_parse_pipeline 并原样返回 ParsedConfig
```

`api.py` 是外部唯一推荐导入点。入口和 Builder 不直接导入内部 normalize/compile 函数。

### `snapshot.py`

函数：

```text
to_serializable_config(config) → dict
save_config_snapshot(config, output_path) → pathlib.Path
```

转换规则：

```text
Path → str
Mapping/list → 递归新容器
普通 YAML 标量 → 原样
其他对象 → TypeError
```

保存使用 `yaml.safe_dump`，明确 `allow_unicode=True` 和稳定字段顺序策略。

### 本阶段完成标准

```text
公共入口只有 load_config
Pipeline 顺序只有一处定义
ParsedConfig 字段明确
YAML 异常转换为 YamlLoadError
快照不包含 Spec/Schema/Context 对象
Override 判别字段仍被拒绝
```

## 30.14 阶段十二：内置扩展

### 本阶段文件

```text
main/parse/builtins/root_schema.py
main/parse/builtins/register.py
main/parse/builtins/modes/train.py
main/parse/builtins/modes/test.py
main/parse/builtins/modes/inference.py
main/parse/builtins/tasks/mlm.py
main/parse/builtins/tasks/qa.py
main/parse/builtins/components/*.py
```

### 本阶段依赖

```text
标准库：仅 dataclasses/typing/pathlib 等创建 Schema 所需模块
项目内：SchemaNode、Spec、Validator、Registry
第三方库：组件 Factory 可以导入 torch；纯 Spec 文件尽量不导入重量级运行模块
```

### 每个文件实现什么

`root_schema.py`：

```text
build_root_schema() → MappingSchema
```

只声明：

```text
version
experiment
run.mode
task.type/task.params
```

Mode 专属字段不放进 Root。

`modes/train.py`：

```text
build_train_mode_spec() → ModeSpec
```

声明 Train 分区、BootstrapDescriptor、Mode Validator 和 Runner 引用。

`tasks/qa.py`：

```text
build_qa_task_spec() → TaskSpec
```

声明 QA 参数、能力要求、Task Validator 和 builders_by_mode。

每个 `components/*.py`：一个文件可登记一类紧密组件，例如：

```text
build_adamw_spec() → ComponentSpec
build_step_lr_spec() → ComponentSpec
```

不要让一个 `components/all.py` 包含所有组件。

### `register.py`

函数：

```text
create_default_services() → ParserServices
```

流程：

```text
创建空 Registry
→ 注册 ModeSpecs
→ 注册 TaskSpecs
→ 注册 ComponentSpecs
→ 创建 deriver_registry
→ build_root_schema
→ 返回 ParserServices
```

注册顺序不影响语义，但应保持稳定以得到可复现的错误和 fingerprint。

### 本阶段完成标准

```text
内置文件只声明扩展，不复制 Pipeline
每个 Spec 都能独立 validate_definition
create_default_services 可重复调用且不共享可变 Registry 状态
QA/MLM 具体名称只出现在 builtins 和任务 Builder
```

## 30.15 阶段十三：入口、Runner 和 Builder 改造

### 本阶段文件

入口：

```text
main/start_train.py
main/start_test.py
main/start_inference.py
```

建议新增运行协议：

```text
main/runtime/context.py
main/runtime/runners/train.py
main/runtime/runners/test.py
main/runtime/runners/inference.py
main/runtime/build.py
```

若项目不希望新增 `runtime/`，可放到现有对应模块，但职责保持相同。

### `BuildContext`：数据类

字段：

```text
parsed_config
services
logger
experiment_paths
device_summary
```

它由 Runner 创建，Task Builder 消费。

### `build_component()`

文件建议：`main/runtime/build.py`。

签名：

```text
build_component(category, node_config, build_context, dependencies=None) → Any
```

流程：

```text
读取 canonical type
→ component_registry.get(category, type)
→ 合并显式 dependencies 与 params
→ 调用 spec.build
→ 返回组件实例
```

不要让 ComponentSpec 从全局变量猜 Tokenizer、Optimizer 等依赖。

### Runner

ModeSpec.runner 指向对应函数：

```text
run_train(build_context, task_builder) → TrainSummary
run_test(build_context, task_builder) → TestSummary
run_inference(build_context, task_builder) → InferenceResult
```

Runner 负责模式流程；Task Builder 负责构造组件。二者不能合并成一个巨型 Builder。

### 入口流程

```text
parse CLI
→ 创建 ConfigOverride
→ load_config(expected_mode=固定模式)
→ 创建实验目录和日志
→ 创建 BuildContext
→ 从 TaskRegistry 取得 TaskSpec
→ task_spec.get_builder(mode_key)
→ mode_spec.get_runner()
→ runner(build_context, builder)
```

入口不拼接 `task_mode` 字符串。

### Builder 改造顺序

```text
1. 新建一个 QA Test Builder，不直接覆盖旧 Builder
2. 参数改为 build_qa_test(context: BuildContext)
3. 使用新嵌套 config 路径
4. 将通用组件构造迁移到 build_component
5. 保留任务依赖注入
6. 返回 Runner 需要的明确结果数据类/字典
7. QA Test 通过后迁移 QA Train
8. 再迁移 MLM
```

### 本阶段完成标准

```text
入口只导入 parse.api，不导入内部 Schema 模块
Runner 根据 mode 选择
Builder 根据 task+mode 选择，但不拼字符串
Builder 不再 strip/casefold type
Builder 不再为 params 补默认值
旧 Builder 在迁移期间仍可回退
```

## 30.16 第一条垂直切片的具体执行链

### 目标和涉及文件

目标：在不切换正式入口的前提下完成 QA Test 新链路。

至少涉及：

```text
一个新 QA Test YAML
builtins/modes/test.py
builtins/tasks/qa.py
builtins/components/ 下 QA 所需 Spec
parse/api.py
新的 QA Test Builder
对应的 Parser/Builder 测试
```

### 实现顺序和每步输出

```text
1. 定义 QA Test YAML
   输出：只含 run.checkpoint、测试 data、test metrics、runtime

2. build_test_mode_spec()
   输出：ModeSpec(key="test")

3. build_qa_task_spec()
   输出：支持 test 的 TaskSpec，包含 test Builder

4. 创建组件 Specs
   输出：Tokenizer/Model/Loss/Dataset/Collator/Metric 的 ComponentSpec

5. create_default_services()
   输出：包含上述 Spec 的 ParserServices

6. load_config(test_yaml, expected_mode="test")
   输出：test_parsed: ParsedConfig

7. 读取 test_parsed.config["run"]["checkpoint"]["train_config"]
   输出：绝对 Path

8. load_config(train_config_path, expected_mode="train", validate_runtime=false)
   输出：train_parsed: ParsedConfig

9. 创建 QA Test BuildContext
   输入：test_parsed、train_parsed、logger、device

10. 调用 qa_task_spec.get_builder("test")
    输出：QA Test 构造结果

11. Builder 构造测试
    检查：Tokenizer/Model/Dataset/Metric 可创建，尚不跑 Batch

12. 单 Batch 测试
    检查：数据字段、模型输出、Loss、Metric 接口匹配

13. 独立 Test Runner
    输出：TestSummary 和指标文件
```

### 双配置所有权

```text
测试配置
→ Dataset 测试 Split
→ DataLoader
→ Metrics
→ Runtime Device
→ Checkpoint path

训练配置
→ Task 类型
→ Tokenizer
→ Model 结构参数
→ Loss 结构

Checkpoint
→ Model 权重
```

若同一字段在两份配置中都出现，必须在 Schema/Validator 中明确禁止或明确优先级，不能由 Builder 临时决定。

### 完成标准

```text
正式 start_test.py 尚可继续使用旧系统
新 QA Test Parser 可独立调用
两份 ParsedConfig 职责明确
Builder 不访问旧扁平路径
完成构造测试、单 Batch 测试和独立 Test
```

## 30.17 新 YAML 的分区到接口映射

```text
experiment
→ Root Schema
→ Experiment Validator

run
→ Root/Mode Schema
→ Bootstrap、Checkpoint/Resume Validator

task
→ TaskSpec
→ Task 参数规范化和 Task Validator

components.tokenizer
→ ComponentSpec(category="tokenizer")

components.model
→ ComponentSpec(category="model")

components.losses
→ SequenceSchema + ComponentSpec(category="loss")

data.dataset
→ ComponentSpec(category="dataset")

data.collator
→ ComponentSpec(category="collator")

data.loaders.<split>
→ Loader Schema + DataLoader Validator

optimization.optimizer
→ ComponentSpec(category="optimizer")

optimization.scheduler
→ ComponentSpec(category="scheduler")

optimization.loop/validation
→ Train Mode Schema 和 Validator

evaluation.metrics.<split>
→ SequenceSchema + ComponentSpec(category="metric")

inference
→ Inference Mode Schema 和 Decoding ComponentSpec

runtime
→ Runtime Schema 和 Runtime Validator
```

每个 YAML 字段必须能回答：

```text
由哪个 Schema 声明
由哪个 Normalizer 处理
由哪个 PathSchema 解析（如果是路径）
由哪个 Validator 校验
由哪个 Builder 消费
```

## 30.18 不应实现的形式

不要实现：

```text
一个包含所有 YAML 字段的 Config 类
一个包含所有任务 if/elif 的 Parser 类
一个巨大的 validate_config 函数
一个按字符串拼接 mode/task 的 Builder Registry
一个字段名到路径规则的全局 if/elif 表
一个直接修改 raw_config 的递归函数
```

理想架构不是“一个万能类”，而是：

```text
少量通用函数
+ 明确的 SchemaNode 类
+ Mode/Task/Component Spec
+ 独立 Registry
+ 注册式校验器
+ 一条固定 Pipeline
```

## 30.19 每个接口实现后的检查方式

每完成一个接口，必须记录：

```text
输入类型
输出类型
是否修改输入
可能抛出的异常
使用的标准库/第三方库
调用者
被调用者
最小测试样例
```

例如完成 `normalize_structure()` 后，应确认：

```text
输入 raw_config 不变
输出是新 dict
未知字段能定位
params=None 仍合法
默认 dict 不共享
Path 尚未被强制解析
```

完成 `resolve_paths()` 后，应确认：

```text
输入 config 不变
输出路径是绝对 pathlib.Path
字段基准正确
推导路径稳定
循环依赖报 PathDependencyError
```

完成 `load_config()` 后，应确认：

```text
五类内置配置都能完成解析
错误来自正确阶段
返回 ParsedConfig
最终 config 可被 Builder 消费
```

## 30.20 推荐的实际开发节奏

每次只做一个最小闭环：

```text
阅读目标接口
→ 明确输入/输出/副作用
→ 实现一个类或一组紧密函数
→ 写最小单元测试
→ 运行测试
→ 阅读磁盘结果
→ 再进入下一接口
```

当前阶段不应一次性实现所有 Schema、所有 Task 和所有 Builder。正确顺序是：

```text
基础节点
→ 小型人工 Schema
→ Schema Merge
→ Registry
→ Bootstrap
→ QA Test
→ QA Train
→ MLM
→ Inference
→ GPT 模拟扩展
```

这样如果某一阶段出现问题，可以确定问题属于：

```text
节点协议
Schema 合并
Spec 注册
Bootstrap
规范化
路径
校验
Builder 适配
```

而不会因为一次修改几十个文件而无法定位。
