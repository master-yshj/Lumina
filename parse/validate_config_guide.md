# `validate_config()` 校验标准

本文档以当前 `main/cfg/example.yaml`、`CUSTOM_DICT` 和已实现组件接口为准，用于重写 `main/parse/parser.py` 中的 `validate_config()`。

## 1. 职责边界

### `check_and_add_config()` 已负责

- 配置整体结构检查；
- 必须显式存在的键检查；
- 未知键检查；
- 值为 `None` 时的默认值填充；
- `losses` 公共结构及其 `params` 模板补全；
- `loss.type` 的最低限度检查，因为它决定使用哪个参数模板。

### `build_config()` 已负责

- 将路径字段转换为绝对 `pathlib.Path`；
- `model_path`、`vocab_file_path`、`dataset_path` 相对于 YAML 目录解析；
- `train_path`、`valid_path`、`test_path` 相对于 `dataset_path` 解析。

### `validate_config()` 负责

- 字符串是否非空；
- 类型是否合法；
- 数值范围是否合法；
- 支持项是否合法；
- 路径指向的目录或文件是否存在且有效；
- 字段之间的约束是否合法；
- Loss 名称、权重和参数的语义是否合法。

`validate_config()` 只校验，不再修改配置。校验成功时返回 `None` 即可。

---

## 2. 通用校验原则

### 2.1 严格排除 `bool`

Python 中 `bool` 是 `int` 的子类，因此整数校验统一使用：

```text
type(value) is int
```

数值校验统一使用：

```text
type(value) in (int, float)
```

不要使用 `isinstance(value, int)` 判断数值，否则 `True` 和 `False` 会被误判为数字。

### 2.2 浮点数必须有限

凡是学习率、权重、Dropout、Gamma、Epsilon 等浮点参数，除类型和范围外，还应使用 `math.isfinite()` 拒绝：

- `NaN`；
- `+Inf`；
- `-Inf`。

因此 `parser.py` 需要导入 `math`，或者从 `math` 导入 `isfinite`。

### 2.3 字符串检查顺序

字符串字段统一按以下顺序检查：

1. 必须是 `str`；
2. `value.strip()` 后不能为空；
3. 需要查询支持项时，使用 `value.casefold()`。

除非后续组件明确需要，否则不要直接修改配置中的原始大小写。

### 2.4 路径检查

`build_config()` 执行后，所有路径字段都应为绝对 `pathlib.Path`。

目录字段统一检查：

1. 值必须是 `pathlib.Path`；
2. 必须是绝对路径；
3. `is_dir()` 必须为 `True`；
4. 如该目录必须包含实现或数据，再检查目录非空。

文件字段统一检查：

1. 值必须是 `pathlib.Path`；
2. 必须是绝对路径；
3. `is_file()` 必须为 `True`；
4. `stat().st_size > 0`；
5. 文本文件至少存在一行非空内容。

### 2.5 错误信息

错误信息应包含完整字段路径和实际值，例如：

```text
config.model.hidden_size 必须为正整数，当前值为：0
```

不要只写“参数错误”，否则新生无法定位 YAML 中的位置。

---

## 3. 顶层实验字段

### `config.name`

- 类型：`str`；
- 去除首尾空白后不能为空。

### `config.seed`

- 类型：严格 `int`；
- 建议允许 `0`，约束为 `seed >= 0`。

说明：随机种子 `0` 是合法值，没有必要要求严格大于零。

### `config.task`

- 类型：非空 `str`；
- `task.casefold()` 必须属于 `SUPPORTED_TASKS`；
- 当前合法值包含 `mlm_pretrain`。

---

## 4. `model` 校验

### `config.model.type`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_MODEL_TYPES`；
- 当前合法值包含 `transformer`。

### `config.model.name`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_MODEL_NAMES`；
- 当前合法值包含 `tinybertformaskedlm`。

### `config.model.model_path`

- 类型：`pathlib.Path`；
- 必须是绝对路径；
- 必须指向存在的目录；
- 目录必须非空。

### 正整数模型参数

以下字段都必须满足：

- 类型严格为 `int`；
- 值大于 `0`。

字段：

- `config.model.hidden_size`；
- `config.model.max_position_embeddings`；
- `config.model.type_vocab_size`；
- `config.model.num_heads`；
- `config.model.intermediate_size`；
- `config.model.num_layers`。

### Dropout 参数

以下字段都必须满足：

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 范围为 `[0, 1)`。

字段：

- `config.model.embedding_dropout`；
- `config.model.attention_dropout`；
- `config.model.hidden_dropout`。

### `config.model.layer_norm_eps`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 必须大于 `0`。

### 模型跨字段约束

必须满足：

```text
hidden_size % num_heads == 0
```

原因：多头注意力需要将隐藏维度平均分配给各注意力头。

还应满足：

```text
dataset.max_seq_length <= model.max_position_embeddings
```

原因：数据集输出的序列长度不能超过位置嵌入可支持的最大长度。

---

## 5. `tokenizer` 校验

### `config.tokenizer.type`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_TOKENIZERS`。

### `config.tokenizer.vocab_file_path`

- 类型：`pathlib.Path`；
- 必须是绝对路径；
- 必须指向存在的文件；
- 文件字节大小必须大于 `0`；
- 至少存在一行非空文本。

特殊 Token 是否齐全、Token 是否重复等内容级校验由 `ExampleTokenizer` 构造函数继续负责，Parser 不重复解析整个词表。

### `config.tokenizer.do_lower_case`

- 类型必须严格为 `bool`。

### `config.tokenizer.strip_accents`

- 类型必须是 `bool` 或 `None`。

### `config.tokenizer.max_input_chars_per_word`

- 类型严格为 `int`；
- 必须大于 `0`。

---

## 6. `dataset` 校验

### `config.dataset.type`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_DATASET`；
- 当前合法值包括 `tiny_zhwiki`、`full_zhwiki`、`base_zhwiki` 等已注册数据集。

### `config.dataset.dataset_path`

- 类型：`pathlib.Path`；
- 必须是绝对路径；
- 必须指向存在的目录；
- 目录必须非空。

### Split 文件

以下三个字段在当前版本中全部必须有效，不受 `train.do_validation` 或 `test.enabled` 影响：

- `config.dataset.train_path`；
- `config.dataset.valid_path`；
- `config.dataset.test_path`。

每个字段都必须满足：

- 类型为 `pathlib.Path`；
- 必须是绝对路径；
- 必须指向存在的普通文件；
- 文件字节大小大于 `0`；
- 至少存在一行非空文本。

说明：

- `valid_path` 和 `test_path` 是当前版本要求提供的数据资产；
- `train.do_validation` 只控制训练期间是否执行验证；
- `test.enabled` 只控制本次运行是否执行测试。

### `config.dataset.max_seq_length`

- 类型严格为 `int`；
- 必须大于 `0`；
- 必须小于等于 `config.model.max_position_embeddings`。

### `config.dataset.mlm_prob`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 推荐范围为 `(0, 1)`。

使用严格开区间是因为：

- `0` 表示不进行任何 MLM 掩码；
- `1` 表示所有候选 Token 都参与掩码，通常不是合理的 MLM 训练设置。

---

## 7. `dataloader` 校验

### `config.dataloader.batch_size`

- 类型严格为 `int`；
- 必须大于 `0`。

### `config.dataloader.shuffle`

- 类型必须严格为 `bool`。

### `config.dataloader.num_workers`

- 类型严格为 `int`；
- 必须大于等于 `0`。

### `config.dataloader.pin_memory`

- 类型必须严格为 `bool`。

### `config.dataloader.drop_last`

- 类型必须严格为 `bool`。

说明：是否根据 CUDA 自动要求 `pin_memory=True` 不属于合法性约束，不应在 Parser 中强制。

---

## 8. `losses` 校验

`check_and_add_config()` 已检查列表结构、公共键、支持类型和 `params` Schema。`validate_config()` 继续检查语义。

### 整体列表

- `config.losses` 必须是非空 `list`；
- 每一项必须是 `dict`；
- Loss 数量不设固定上限。

虽然结构阶段已检查过，`validate_config()` 可以依赖规范化结果，不必重复所有键存在性检查。

### `losses[i].name`

- 类型：非空 `str`；
- 建议对 `strip()` 后的原始名称做唯一性检查；
- 更稳妥的方式是使用 `casefold()` 后检查唯一性，避免 `MLM_Loss` 与 `mlm_loss` 在日志中语义重复。

### `losses[i].type`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_LOSSES`；
- 必须在 `LOSS_PARAMS_CUSTOM_DICT` 中存在对应参数模板。

这部分即使结构阶段已检查，也可以保留轻量断言式校验，避免未来绕过 `build_config()` 直接调用 `validate_config()` 时出现不清楚的错误。

### `losses[i].weight`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 必须大于等于 `0`。

整体还必须满足：

```text
至少一个 Loss 的 weight > 0
```

否则总损失恒为零，无法训练。

### `CrossEntropyLoss.params.ignore_index`

当 `loss.type.casefold() == "crossentropyloss"` 时：

- `ignore_index` 类型严格为 `int`；
- 当前 MLM 链路必须等于 `-100`。

原因：当前 `MLMCollator` 的非监督位置标签使用 `-100`，必须与交叉熵忽略标签一致。

未来增加其他 Loss 时，在此处按 `loss.type.casefold()` 增加对应参数值校验；不要让所有 Loss 共用 `ignore_index` 规则。

---

## 9. `optimizer` 校验

### `config.optimizer.type`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_OPTIMIZERS`；
- 当前合法值包含 `adamw`。

### `config.optimizer.lr`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 必须大于 `0`。

### `config.optimizer.weight_decay`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 必须大于等于 `0`。

### `config.optimizer.betas`

在配置层中必须满足：

- 类型为 `list`；
- 长度恰好为 `2`；
- 每个元素类型为 `int` 或 `float`，但不能是 `bool`；
- 每个元素数值有限；
- 每个元素满足 `[0, 1)`。

建议额外满足：

```text
beta1 < beta2
```

这符合 AdamW 的常见设置，但 PyTorch 并不要求该关系。为了避免无必要限制，可以暂不强制，只校验两个值都位于 `[0, 1)`。

入口创建 `AdamW` 时再将列表转换为元组。

### `config.optimizer.eps`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 必须大于 `0`。

---

## 10. `scheduler` 校验

### `config.scheduler.type`

- 类型：非空 `str`；
- `casefold()` 后必须属于 `SUPPORTED_SCHEDULERS`；
- 当前合法值包含 `steplr`。

### `config.scheduler.interval`

- 类型：非空 `str`；
- 当前只允许：

```text
step
```

此字段决定 Trainer 在每次 Optimizer 更新后调用 `scheduler.step()`。

### `config.scheduler.step_size`

- 类型严格为 `int`；
- 必须大于 `0`。

### `config.scheduler.gamma`

- 类型为 `int` 或 `float`，但不能是 `bool`；
- 数值有限；
- 必须大于等于 `0`。

可以允许 `gamma == 0`，它会在首次衰减时将学习率降为零；如果不希望这种实验配置，也可以将项目规则收紧为 `gamma > 0`。建议当前与 PyTorch 能力保持一致，允许零。

### `config.scheduler.last_epoch`

- 类型严格为 `int`；
- 必须大于等于 `-1`。

当前新训练配置应使用 `-1`。恢复训练以后优先恢复 Scheduler 的 `state_dict()`，而不是只修改该字段。

---

## 11. `train` 校验

### 正整数训练参数

以下字段必须是严格 `int` 且大于 `0`：

- `config.train.total_iters`；
- `config.train.accumulation_steps`；
- `config.train.log_every_iters`。

### `config.train.do_validation`

- 类型必须严格为 `bool`。

### 建议的训练跨字段约束

建议满足：

```text
log_every_iters <= total_iters
```

否则整个训练期间可能一次周期日志都不输出。该约束可作为警告而非硬错误；如果当前没有警告系统，可以暂不强制。

---

## 12. `valid` 校验

以下字段必须是严格 `int` 且大于 `0`：

- `config.valid.valid_every_iters`；
- `config.valid.early_stop_patience`；
- `config.valid.validation_rounds_before_save_ckpt`。

### 验证跨字段约束

当 `config.train.do_validation is True` 时，建议满足：

```text
valid_every_iters <= total_iters
```

否则训练结束前不会触发一次验证。

如果 `do_validation is False`，`valid` 配置仍需类型和值合法，但 Trainer 不执行验证。

`valid_path` 在当前版本中始终要求存在，与 `do_validation` 无关。

---

## 13. `test` 校验

### `config.test.enabled`

- 类型必须严格为 `bool`。

当前测试链路尚未接入时，建议增加当前阶段约束：

```text
test.enabled 必须为 False
```

这样用户设置为 `True` 时会立即得到明确错误，而不是配置被静默忽略。

当测试入口实现后，删除“必须为 False”的临时限制，只保留布尔类型校验。

`test_path` 在当前版本中始终要求存在，与 `test.enabled` 无关。

---

## 14. `cuda` 校验

### `config.cuda.device`

- 类型：非空 `str`；
- 当前允许：
  - `cpu`；
  - `cuda`；
  - `cuda:N`，其中 `N` 是大于等于 `0` 的十进制整数。

建议使用明确逻辑或正则表达式判断，不要只检查字符串是否以 `cuda` 开头。

如果设备为 `cuda` 或 `cuda:N`，还应检查：

- `torch.cuda.is_available()` 为 `True`；
- 若为 `cuda:N`，则 `N < torch.cuda.device_count()`。

是否在 Parser 中检查硬件环境有两种边界：

- 严格启动校验：在这里检查，尽早报错；
- 纯配置校验：只检查字符串格式，硬件可用性交给入口。

建议当前采用严格启动校验，因为该配置将立即用于本机训练。

### `config.cuda.gpu_nums`

- 类型严格为 `int`；
- 必须大于 `0`；
- 当前最小单设备链路要求：

```text
gpu_nums == 1
```

### 设备跨字段约束

当前非 DDP 阶段：

- `device == "cpu"` 时，`gpu_nums` 仍按当前 YAML 约定保持 `1`，表示单设备执行，而不是实际使用一张 GPU；
- `device` 为 CUDA 时，`gpu_nums` 必须为 `1`；
- 后续接入 DDP 后再重新定义 `gpu_nums` 与 `WORLD_SIZE` 的关系。

---

## 15. 推荐的实现顺序

按以下顺序编写 `validate_config()`，每完成一组再继续：

1. `name`、`seed`、`task`；
2. `model.type`、`model.name`、`model_path`；
3. 模型数值参数和模型内部跨字段约束；
4. `tokenizer`；
5. `dataset` 和四个数据路径字段；
6. `dataloader`；
7. `losses`；
8. `optimizer`；
9. `scheduler`；
10. `train`；
11. `valid`；
12. `test`；
13. `cuda`；
14. 所有跨组约束。

---

## 16. 当前必须实现与可选约束

### 当前必须实现

- 所有字段的基础类型；
- 所有字符串非空；
- 所有支持项检查；
- 所有数值的有限性和基础范围；
- 所有路径存在性、文件/目录类型和非空性；
- `hidden_size % num_heads == 0`；
- `max_seq_length <= max_position_embeddings`；
- Loss 名称唯一；
- Loss 权重非负且至少一个大于零；
- `CrossEntropyLoss.ignore_index == -100`；
- Optimizer Betas 是长度为 2 的列表且元素位于 `[0, 1)`；
- Scheduler Interval 当前等于 `step`；
- `test.enabled` 当前必须为 `False`；
- `gpu_nums == 1`。

### 可以暂不强制

- `beta1 < beta2`；
- `log_every_iters <= total_iters`；
- `valid_every_iters <= total_iters`；
- CUDA 硬件可用性检查（如果决定交给入口）；
- Dataset 文件内部 JSONL Schema 校验；
- 词表特殊 Token 和重复 Token 校验。

---

## 17. 完成标准

重写后的 `validate_config()` 应满足：

- 不再访问旧 YAML 字段；
- 不再出现 `modules`；
- 不再访问顶层 `dataset_path`；
- 不再访问 `train.dataset`、`train.loss`、`train.optimizer`、`train.scheduler`、`train.embedding`；
- 不再访问 `valid.dataset`、`test.dataset`；
- 不再访问 `cuda.num_workers`；
- 校验失败时抛出包含完整字段路径的 `ValueError`；
- 文件不存在时可以统一使用 `ValueError`，也可以使用 `FileNotFoundError`，但项目内应保持一致；
- 校验过程不修改 `config_dict`；
- 校验成功时正常返回 `None`。
