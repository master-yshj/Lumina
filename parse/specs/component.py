import pathlib

from collections.abc import Mapping, Callable
from dataclasses import dataclass
from typing import Any
from copy import deepcopy

from Inference.decoding.registry import DECODING_REGISTRY
from metrics.registry import METRIC_REGISTRY
from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from tasks.registry import TASK_BUILDER_REGISTRY
# AI-CHANGE: 可视化方法运行 Registry 接入(AI 编写, 待人工审查)。
# 只引入方法级 Registry,不涉及任何任务 Builder,模型与数据集声明另行设计。
from visualize.registry import VISUALIZER_REGISTRY
from visualize.methods.loss_landscape import LossLandscapeVisualizer
from parse.errors import UnknownFieldError, TypeMismatchError
from parse.schema.nodes import MappingSchema, SchemaNode, PathSchema, ScalarSchema, EnumSchema, SequenceSchema, RegistryKeySchema, PolymorphicSchema
from parse.specs.validator import ValidatorSpec
from parse.schema.merge import merge_schema


@dataclass(frozen=True)
class BootstrapDescriptor:
    path_pattern: tuple[str, ...]
    category: str
    required: bool = True

    def __post_init__(self):
        if not isinstance(self.path_pattern, tuple):
            raise TypeError(f"path_pattern: {self.path_pattern} 必须为tuple类型的喵!")
        if not self.path_pattern:
            raise ValueError("path_pattern 不能为空喵!")
        for part in self.path_pattern:
            if not isinstance(part, str):
                raise TypeError(f"path_pattern: {self.path_pattern} 中的元素必须为str类型喵!")
            if not part.strip():
                raise ValueError(f"path_pattern: {self.path_pattern} 中的元素不能为空喵!")
        if not isinstance(self.category, str):
            raise TypeError(f"category: {self.category} 必须为str类型喵!")
        if not self.category.strip():
            raise ValueError("category 不能为空喵!")
        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型喵!")


@dataclass(frozen=True)
class ComponentSpec:
    category: str                                       # 表示组件所属类别
    key: str                                            # 表示组件的规范名称，也就是 canonical key
    aliases: tuple[str, ...]                            # 表示允许用户在 YAML 中使用的别名，比如tiny_bert_mlm也可以写成TinyBertMlm
    params_schema: MappingSchema                        # 表示该组件 params 字段的参数结构规则
    default_params: Mapping[str, Any]                   # 表示组件默认参数
    capabilities: frozenset[str]                        # 表示组件提供的能力集合，只用来描述组件的能力
    normalizers: tuple[Callable, ...]                   # 表示组件参数规范化之后还需要执行的组件级参数转换函数
    validators: tuple[ValidatorSpec, ...]               # 表示该组件自己的参数或配置校验器
    factory: Callable | None                            # 表示由规范化参数构造真实运行组件的函数
    schema_contribution: SchemaNode | None              # 表示该组件对最终总 Schema 提供的额外 Schema 贡献
    bootstrap_descriptors: tuple[Any, ...] = ()         # 表示Bootstrap阶段需要扫描的动态组件节点描述

    def __post_init__(self):
        if not isinstance(self.category, str):
            raise TypeError(f"category: {self.category} 必须为str类型的喵! 现在的type: {type(self.category)}")
        if not self.category.strip():
            raise ValueError(f"category 不能为空喵!")
        if not isinstance(self.key, str):
            raise TypeError(f"key: {self.key} 必须为str类型的喵! 现在的type: {type(self.key)}")
        if not self.key.strip():
            raise ValueError(f"key 不能为空喵!")
        if not isinstance(self.aliases, tuple):
            raise TypeError(f"aliases: {self.aliases} 必须为tuple类型的喵! 现在的type: {type(self.aliases)}")
        normalized_aliases = []
        normalized_key = self.key.strip().casefold()
        for alias in self.aliases:
            if not isinstance(alias, str):
                raise TypeError(f"aliases: {self.aliases} alias: {alias} 必须为str类型的喵! 现在的type: {type(alias)}")
            normalized_alias = alias.strip().casefold()
            if not normalized_alias:
                raise ValueError(f"aliases: {self.aliases} 中的alias不能为空喵!")
            if normalized_alias in normalized_aliases:
                raise ValueError(f"aliases: {self.aliases} 中的alias在规范化后: {normalized_alias} 不能重复喵!")
            else:
                normalized_aliases.append(normalized_alias)
            if normalized_key in normalized_aliases:
                raise ValueError(f"aliases: {self.aliases} normalized_alias: {normalized_alias} 与 key: {self.key} normalized_key: {normalized_key} 重复了喵!")
        if not isinstance(self.params_schema, MappingSchema):
            raise TypeError(f"params_schema: {self.params_schema} 必须为MappingSchema类型的喵! 现在的type: {type(self.params_schema)}")
        if not isinstance(self.default_params, Mapping):
            raise TypeError(f"default_params: {self.default_params} 必须为Mapping类型的喵! 现在的type: {type(self.default_params)}")
        different_keys = set(self.default_params.keys()) - set(self.params_schema.fields.keys())
        if different_keys:
            raise UnknownFieldError(f"default_params: {self.default_params} 与 params_schema: {self.params_schema} 出现了不同字段: {different_keys} 喵!")

        if not isinstance(self.capabilities, frozenset):
            raise TypeError(f"capabilities: {self.capabilities} 必须为frozenset类型的喵! 现在的type: {type(self.capabilities)}")
        for capability in self.capabilities:
            if not isinstance(capability, str):
                raise TypeError(f"capabilities: {self.capabilities} capability: {capability} 必须为str类型的喵! 现在的type: {type(capability)}")
            if not capability.strip():
                raise ValueError(f"capabilities: {self.capabilities} 中的每个元素都不能为空喵!")
        if not isinstance(self.normalizers, tuple):
            raise TypeError(f"normalizers: {self.normalizers} 必须为tuple类型的喵! 现在的type: {type(self.normalizers)}")
        for normalizer in self.normalizers:
            if not callable(normalizer):
                raise TypeError(f"normalizers: {self.normalizers} normalizer: {normalizer} 必须为可调用对象喵! 现在的type: {type(normalizer)}")
        if not isinstance(self.validators, tuple):
            raise TypeError(f"validators: {self.validators} 必须为tuple类型的喵! 现在的type: {type(self.validators)}")
        for validator in self.validators:
            if not isinstance(validator, ValidatorSpec):
                raise TypeError(f"validators: {self.validators} validator: {validator} 必须为ValidatorSpec类型的喵! 现在的type: {type(validator)}")
        if self.factory is not None and not callable(self.factory):
            raise TypeError(f"factory: {self.factory} 必须为None或者可调用喵! 现在的type: {type(self.factory)}")
        if self.schema_contribution is not None and not isinstance(self.schema_contribution, SchemaNode):
            raise TypeError(f"schema_contribution: {self.schema_contribution} 必须为None或者SchemaNode类型的喵! 现在的type: {type(self.schema_contribution)}")
        if not isinstance(self.bootstrap_descriptors, tuple):
            raise TypeError(f"bootstrap_descriptors: {self.bootstrap_descriptors} 必须为tuple类型的喵! 现在的type: {type(self.bootstrap_descriptors)}")
        for descriptor in self.bootstrap_descriptors:
            if not hasattr(descriptor, "path_pattern") or not hasattr(descriptor, "category"):
                raise TypeError(f"bootstrap_descriptors: {self.bootstrap_descriptors} 中的元素必须为BootstrapDescriptor类型的喵!")
        for field_name, default_value in self.default_params.items():
            field_schema = self.params_schema.fields[field_name]

    def _recursive_merge(self, user_mapping, default_mapping):
        result = deepcopy(default_mapping)
        for field_name, user_value in user_mapping.items():
            if not field_name in default_mapping:
                result[field_name] = deepcopy(user_value)
            else:
                default_value = default_mapping[field_name]
                if isinstance(user_value, Mapping) and isinstance(default_value, Mapping):
                    result[field_name] = self._recursive_merge(
                        user_value, default_value
                    )
                else:
                    result[field_name] = deepcopy(user_value)
        return result

    def normalize_params(self, raw_params, path, context):
        copy_default_params = deepcopy(self.default_params)
        if raw_params is None:
            params_before_schema = copy_default_params
        elif isinstance(raw_params, Mapping):
            merged_params = self._recursive_merge(raw_params, copy_default_params)
            params_before_schema = merged_params
        else:
            raise TypeMismatchError(f"raw_params: {raw_params} 必须为Mapping类型的喵! 现在的type: {type(raw_params)}")
        normalized_params = self.params_schema.normalize(
            value=params_before_schema,
            path=path,
            context=context,
        )
        for normalizer in self.normalizers:
            normalized_params = normalizer(normalized_params, path, context)
        return normalized_params


# TOKENIZER
TOKENIZER_CANONICAL_MAP = TOKENIZER_REGISTRY.canonical_map
TOKENIZER_LOOKUP_MAP = TOKENIZER_REGISTRY.lookup_map

ExampleTokenizer = TOKENIZER_CANONICAL_MAP["exampletokenizer"]
QATokenizer = TOKENIZER_CANONICAL_MAP["qatokenizer"]

ExampleTokenizer_aliases = []
QATokenizer_aliases = []
for alias, canonical_key in TOKENIZER_LOOKUP_MAP.items():
    if canonical_key == 'exampletokenizer':
        ExampleTokenizer_aliases.append(alias) if alias != 'exampletokenizer' else None
    elif canonical_key == 'qatokenizer':
        QATokenizer_aliases.append(alias) if alias != 'qatokenizer' else None
ExampleTokenizer_aliases = tuple(ExampleTokenizer_aliases)
QATokenizer_aliases = tuple(QATokenizer_aliases)


# EXAMPLE_TOKENIZER_PARAMS_SCHEMA
EXAMPLE_TOKENIZER_PARAMS_SCHEMA = MappingSchema(
    fields={
        "vocab_file_path": PathSchema(
            base=pathlib.Path("."),
            kind="file",
            allowed_suffixes=(".txt", ".json", ".jsonl"),
            must_exist=True,
        ),
        "do_lower_case": ScalarSchema(
            python_types=(bool,),
            default=True,
        ),
        "strip_accents": ScalarSchema(
            python_types=(bool, type(None)),
            allow_none=True,
            default=None,
        ),
        "max_input_chars_per_word": ScalarSchema(
            python_types=(int,),
            default=100
        )
    },
    allow_unknown=False,
    allow_none=False,
)

EXAMPLE_TOKENIZER_SPEC = ComponentSpec(
    category="tokenizer",
    key="exampletokenizer",
    aliases=ExampleTokenizer_aliases,
    params_schema=EXAMPLE_TOKENIZER_PARAMS_SCHEMA,
    default_params={
        "do_lower_case": True,
        "strip_accents": None,
        "max_input_chars_per_word": 100,
    },
    capabilities=frozenset(
        name
        for name in dir(ExampleTokenizer)
        if callable(getattr(ExampleTokenizer, name))
        and not name.startswith("_")),
    normalizers=(),
    validators=(),
    factory=ExampleTokenizer,
    schema_contribution=None,
)


#QA_TOKENIZER_PARAMS_SCHEMA
QA_TOKENIZER_PARAMS_SCHEMA = MappingSchema(
    fields={
        "vocab_file_path": PathSchema(
            base=pathlib.Path("."),
            kind="file",
            allowed_suffixes=(".txt", ".json", ".jsonl"),
            must_exist=True,
        ),
        "do_lower_case": ScalarSchema(
            python_types=(bool,),
            default=True,
        ),
        "strip_accents": ScalarSchema(
            python_types=(bool, type(None)),
            allow_none=True,
            default=None,
        ),
        "max_input_chars_per_word": ScalarSchema(
            python_types=(int,),
            default=100
        )
    },
    allow_unknown=False,
    allow_none=False,
)

QA_TOKENIZER_SPEC = ComponentSpec(
    category="tokenizer",
    key="qatokenizer",
    aliases=QATokenizer_aliases,
    params_schema=QA_TOKENIZER_PARAMS_SCHEMA,
    default_params={
        "do_lower_case": True,
        "strip_accents": None,
        "max_input_chars_per_word": 100,
    },
    capabilities=frozenset(
        name
        for name in dir(QATokenizer)
        if callable(getattr(QATokenizer, name))
        and not name.startswith("_")),
    normalizers=(),
    validators=(),
    factory=QATokenizer,
    schema_contribution=None,
)

# MODEL
MODEL_CANONICAL_MAP = MODEL_REGISTRY.canonical_map
MODEL_LOOKUP_MAP = MODEL_REGISTRY.lookup_map

TinyBertForMaskedLM = MODEL_CANONICAL_MAP["tinybertformaskedlm"]
TinyBertForQuestionAnswering = MODEL_CANONICAL_MAP["tinybertforquestionanswering"]

TinyBertForMaskedLM_aliases = []
TinyBertForQuestionAnswering_aliases = []
for alias, canonical_key in MODEL_LOOKUP_MAP.items():
    if canonical_key == 'tinybertformaskedlm':
        TinyBertForMaskedLM_aliases.append(alias) if alias != 'tinybertformaskedlm' else None
    elif canonical_key == 'tinybertforquestionanswering':
        TinyBertForQuestionAnswering_aliases.append(alias) if alias != 'tinybertforquestionanswering' else None
TinyBertForMaskedLM_aliases = tuple(TinyBertForMaskedLM_aliases)
TinyBertForQuestionAnswering_aliases = tuple(TinyBertForQuestionAnswering_aliases)

# TinyBertForMaskedLM_PARAMS_SCHEMA
TinyBertForMaskedLM_PARAMS_SCHEMA = MappingSchema(
    fields={
        # PARSE_REFACTOR_DEMO: 保留旧版 model_path 配置语义；该字段属于
        # Model params，并由 PathSchema 转换为 pathlib.Path 后统一绝对化。
        # 未在 YAML 中显式给出时回落到默认实现目录，避免 normalized 出现 MISSING
        # 导致快照落盘失败。
        "model_path": PathSchema(
            base="config_dir",
            kind="directory",
            required=False,
        ),
        "hidden_size": ScalarSchema(
            python_types=(int,),
            default=128,
        ),
        "max_position_embeddings": ScalarSchema(
            python_types=(int,),
            default=128,
        ),
        "type_vocab_size": ScalarSchema(
            python_types=(int,),
            default=2,
        ),
        "num_heads": ScalarSchema(
            python_types=(int,),
            default=4,
        ),
        "intermediate_size": ScalarSchema(
            python_types=(int,),
            default=512,
        ),
        "num_layers": ScalarSchema(
            python_types=(int,),
            default=2,
        ),
        "embedding_dropout": ScalarSchema(
            python_types=(float,),
            default=0.1,
        ),
        "attention_dropout": ScalarSchema(
            python_types=(float,),
            default=0.1,
        ),
        "hidden_dropout": ScalarSchema(
            python_types=(float,),
            default=0.1,
        ),
        "layer_norm_eps": ScalarSchema(
            python_types=(float,),
            default=1e-12,
        ),
    },
    allow_unknown=False,
    allow_none=False,
)

# TinyBertForQuestionAnswering_PARAMS_SCHEMA
TinyBertForQuestionAnswering_PARAMS_SCHEMA = MappingSchema(
    fields={
        # PARSE_REFACTOR_DEMO: QA Builder 与旧版配置同样读取 model_path，
        # 因此两个 TinyBert Model Spec 必须保持一致的路径参数契约。
        # 未显式给出时回落到默认实现目录，避免 normalized 出现 MISSING。
        "model_path": PathSchema(
            base="config_dir",
            kind="directory",
            required=False,
        ),
        "hidden_size": ScalarSchema(
            python_types=(int,),
            default=128,
        ),
        "max_position_embeddings": ScalarSchema(
            python_types=(int,),
            default=128,
        ),
        "type_vocab_size": ScalarSchema(
            python_types=(int,),
            default=2,
        ),
        "num_heads": ScalarSchema(
            python_types=(int,),
            default=4,
        ),
        "intermediate_size": ScalarSchema(
            python_types=(int,),
            default=512,
        ),
        "num_layers": ScalarSchema(
            python_types=(int,),
            default=2,
        ),
        "embedding_dropout": ScalarSchema(
            python_types=(float,),
            default=0.1,
        ),
        "attention_dropout": ScalarSchema(
            python_types=(float,),
            default=0.1,
        ),
        "hidden_dropout": ScalarSchema(
            python_types=(float,),
            default=0.1,
        ),
        "layer_norm_eps": ScalarSchema(
            python_types=(float,),
            default=1e-12,
        ),
    },
    allow_unknown=False,
    allow_none=False,
)

TinyBertForMaskedLM_SPEC = ComponentSpec(
    category="model",
    key="tinybertformaskedlm",
    aliases=TinyBertForMaskedLM_aliases,
    params_schema=TinyBertForMaskedLM_PARAMS_SCHEMA,
    default_params={
        "hidden_size": 128,
        "max_position_embeddings": 128,
        "type_vocab_size": 2,
        "num_heads": 4,
        "intermediate_size": 512,
        "num_layers": 2,
        "embedding_dropout": 0.1,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "layer_norm_eps": 1e-12,
    },
    capabilities=frozenset(
        name
        for name in dir(TinyBertForMaskedLM)
        if callable(getattr(TinyBertForMaskedLM, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=TinyBertForMaskedLM,
    schema_contribution=None,
)

TinyBertForQuestionAnswering_SPEC = ComponentSpec(
    category="model",
    key="tinybertforquestionanswering",
    aliases=TinyBertForQuestionAnswering_aliases,
    params_schema=TinyBertForQuestionAnswering_PARAMS_SCHEMA,
    default_params={
        "hidden_size": 128,
        "max_position_embeddings": 128,
        "type_vocab_size": 2,
        "num_heads": 4,
        "intermediate_size": 512,
        "num_layers": 2,
        "embedding_dropout": 0.1,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "layer_norm_eps": 1e-12,
    },
    capabilities=frozenset(
        name
        for name in dir(TinyBertForQuestionAnswering)
        if callable(getattr(TinyBertForQuestionAnswering, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=TinyBertForQuestionAnswering,
    schema_contribution=None,
)

# LOSS
LOSS_CANONICAL_MAP = LOSS_REGISTRY.canonical_map
LOSS_LOOKUP_MAP = LOSS_REGISTRY.lookup_map

CrossEntropyLoss = LOSS_CANONICAL_MAP["crossentropyloss"]

CrossEntropyLoss_aliases = []

for alias, canonical_key in LOSS_LOOKUP_MAP.items():
    if canonical_key == "crossentropyloss":
        CrossEntropyLoss_aliases.append(alias) if alias != "crossentropyloss" else None

CrossEntropyLoss_aliases = tuple(CrossEntropyLoss_aliases)

# CrossEntropyLoss_PARAMS_SCHEMA
CrossEntropyLoss_PARAMS_SCHEMA = MappingSchema(
    fields={
        "ignore_index": ScalarSchema(
            python_types=(int,),
            default=-100,
        ),
    },
    allow_unknown=False,
    allow_none=True,
)

CrossEntropyLoss_SPEC = ComponentSpec(
    category="loss",
    key="crossentropyloss",
    aliases=CrossEntropyLoss_aliases,
    params_schema=CrossEntropyLoss_PARAMS_SCHEMA,
    default_params={
        "ignore_index": -100
    },
    capabilities=frozenset(
        name
        for name in dir(CrossEntropyLoss)
        if callable(getattr(CrossEntropyLoss, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=CrossEntropyLoss,
    schema_contribution=None,
)

# OPTIMIZER
OPTIMIZER_CANONICAL_MAP = OPTIMIZER_REGISTRY.canonical_map
OPTIMIZER_LOOKUP_MAP = OPTIMIZER_REGISTRY.lookup_map

AdamW = OPTIMIZER_CANONICAL_MAP["adamw"]

AdamW_aliases = []
for alias, canonical_key in OPTIMIZER_LOOKUP_MAP.items():
    if canonical_key == "adamw":
        AdamW_aliases.append(alias) if alias != "adamw" else None

AdamW_aliases = tuple(AdamW_aliases)

# PARSE_REFACTOR_DEMO: YAML 没有 tuple 类型，betas 从 YAML 读取为 list；
# 规范化后转换为 tuple，以满足 AdamW 的参数契约。
def _normalize_betas(value):
    return tuple(value)


AdamW_PARAMS_SCHEMA = MappingSchema(
    fields={
        "lr": ScalarSchema(
            python_types=(float,int,),
            default=1e-3,
        ),
        "weight_decay": ScalarSchema(
            python_types=(float, int),
            default=0.01,
        ),
        "betas": ScalarSchema(
            python_types=(list, tuple),
            default=(0.9, 0.999),
            normalizer=_normalize_betas,
        ),
        "eps": ScalarSchema(
            python_types=(float, int),
            default=1e-8,
        ),
    },
    allow_unknown=False,
    allow_none=True,
)

AdamW_SPEC = ComponentSpec(
    category="optimizer",
    key="adamw",
    aliases=AdamW_aliases,
    params_schema=AdamW_PARAMS_SCHEMA,
    default_params={
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    },
    capabilities=frozenset(
        name
        for name in dir(AdamW)
        if callable(getattr(AdamW, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=AdamW,
    schema_contribution=None,
)

# SCHEDULER
SCHEDULER_CANONICAL_MAP = SCHEDULER_REGISTRY.canonical_map
SCHEDULER_LOOKUP_MAP = SCHEDULER_REGISTRY.lookup_map

StepLR = SCHEDULER_CANONICAL_MAP["steplr"]

StepLR_aliases = []
for alias, canonical_key in SCHEDULER_LOOKUP_MAP.items():
    if canonical_key == "steplr":
        StepLR_aliases.append(alias) if alias != "steplr" else None

StepLR_aliases = tuple(StepLR_aliases)

# StepLR_PARAMS_SCHEMA
StepLR_PARAMS_SCHEMA = MappingSchema(
    fields={
        "step_size": ScalarSchema(
            python_types=(int,),
            default=1000,
        ),
        "gamma": ScalarSchema(
            python_types=(float,int),
            default=0.1,
        ),
        "last_epoch": ScalarSchema(
            python_types=(int,),
            default=-1,
        )
    },
    allow_unknown=False,
    allow_none=True,
)

StepLR_SPEC = ComponentSpec(
    category="scheduler",
    key="steplr",
    aliases=StepLR_aliases,
    params_schema=StepLR_PARAMS_SCHEMA,
    default_params={
        "step_size": 1000,
        "gamma": 0.1,
        "last_epoch": -1,
    },
    capabilities=frozenset(
        name
        for name in dir(StepLR)
        if callable(getattr(StepLR, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=StepLR,
    schema_contribution=None,
)

WarmupScheduler = SCHEDULER_CANONICAL_MAP["warmup"]

WarmupScheduler_aliases = []
for alias, canonical_key in SCHEDULER_LOOKUP_MAP.items():
    if canonical_key == "warmup":
        WarmupScheduler_aliases.append(alias) if alias != "warmup" else None

WarmupScheduler_aliases = tuple(WarmupScheduler_aliases)

# WarmupScheduler_PARAMS_SCHEMA
WarmupScheduler_PARAMS_SCHEMA = MappingSchema(
    fields={
        "warmup_steps": ScalarSchema(
            python_types=(int,),
            default=1000,
        ),
        "base_scheduler_type": ScalarSchema(
            python_types=(str,),
            default="steplr",
        ),
        "base_scheduler_params": ScalarSchema(
            python_types=(dict,),
            default={},
        ),
    },
    allow_unknown=False,
    allow_none=True,
)

WarmupScheduler_SPEC = ComponentSpec(
    category="scheduler",
    key="warmup",
    aliases=WarmupScheduler_aliases,
    params_schema=WarmupScheduler_PARAMS_SCHEMA,
    default_params={
        "warmup_steps": 1000,
        "base_scheduler_type": "steplr",
        "base_scheduler_params": {},
    },
    capabilities=frozenset(
        name
        for name in dir(WarmupScheduler)
        if callable(getattr(WarmupScheduler, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=WarmupScheduler,
    schema_contribution=None,
)

# Metric
METRIC_CANONICAL_MAP = METRIC_REGISTRY.canonical_map
METRIC_LOOKUP_MAP = METRIC_REGISTRY.lookup_map

TokenAccuracy = METRIC_CANONICAL_MAP["tokenaccuracy"]
ExactMatch = METRIC_CANONICAL_MAP["exactmatch"]
CharacterF1 = METRIC_CANONICAL_MAP["characterf1"]

TokenAccuracy_aliases = []
ExactMatch_aliases = []
CharacterF1_aliases = []

for alias, canonical_key in METRIC_LOOKUP_MAP.items():
    if canonical_key == "tokenaccuracy":
        TokenAccuracy_aliases.append(alias) if alias != "tokenaccuracy" else None
    elif canonical_key == "exactmatch":
        ExactMatch_aliases.append(alias) if alias != "exactmatch" else None
    elif canonical_key == "characterf1":
        CharacterF1_aliases.append(alias) if alias != "characterf1" else None

TokenAccuracy_aliases = tuple(TokenAccuracy_aliases)
ExactMatch_aliases = tuple(ExactMatch_aliases)
CharacterF1_aliases = tuple(CharacterF1_aliases)

# TokenAccuracy_PARAMS_SCHEMA
TokenAccuracy_PARAMS_SCHEMA = MappingSchema(
    fields={
        "ignore_index": ScalarSchema(
            python_types=(int,),
            default=-100,
        )
    },
    allow_unknown=False,
    allow_none=True,
)

# ExactMatch_PARAMS_SCHEMA
ExactMatch_PARAMS_SCHEMA = MappingSchema(
    fields={

    },
    allow_unknown=False,
    allow_none=True,
)

# CharacterF1_PARAMS_SCHEMA
CharacterF1_PARAMS_SCHEMA = MappingSchema(
    fields={

    },
    allow_unknown=False,
    allow_none=True,
)

TokenAccuracy_SPEC = ComponentSpec(
    category="metric",
    key="tokenaccuracy",
    aliases=TokenAccuracy_aliases,
    params_schema=TokenAccuracy_PARAMS_SCHEMA,
    default_params={
        "ignore_index": -100,
    },
    capabilities=frozenset(
        name
        for name in dir(TokenAccuracy)
        if callable(getattr(TokenAccuracy, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=TokenAccuracy,
    schema_contribution=None,
)

ExactMatch_SPEC = ComponentSpec(
    category="metric",
    key="exactmatch",
    aliases=ExactMatch_aliases,
    params_schema=ExactMatch_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(ExactMatch)
        if callable(getattr(ExactMatch, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=ExactMatch,
    schema_contribution=None,
)

CharacterF1_SPEC = ComponentSpec(
    category="metric",
    key="characterf1",
    aliases=CharacterF1_aliases,
    params_schema=CharacterF1_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(CharacterF1)
        if callable(getattr(CharacterF1, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=CharacterF1,
    schema_contribution=None,
)

# DECODING
DECODING_CANONICAL_MAP = DECODING_REGISTRY.canonical_map
DECODING_LOOKUP_MAP = DECODING_REGISTRY.lookup_map

TopKSamplingStrategy = DECODING_CANONICAL_MAP["topksampling"]
QASpanTopKDecodingStrategy = DECODING_CANONICAL_MAP["qaspantopkdecoding"]

TopKSamplingStrategy_aliases = []
for alias, canonical_key in DECODING_LOOKUP_MAP.items():
    if canonical_key == "topksampling":
        TopKSamplingStrategy_aliases.append(alias) if alias != "topksampling" else None
TopKSamplingStrategy_aliases = tuple(TopKSamplingStrategy_aliases)

# TopKSamplingStrategy_PARAMS_SCHEMA
TopKSamplingStrategy_PARAMS_SCHEMA = MappingSchema(
    fields={
        "top_k": ScalarSchema(
            python_types=(int,),
            default=5,
        )
    },
    allow_unknown=False,
    allow_none=True,
)

TopKSamplingStrategy_SPEC = ComponentSpec(
    category="decoding",
    key="topksampling",
    aliases=TopKSamplingStrategy_aliases,
    params_schema=TopKSamplingStrategy_PARAMS_SCHEMA,
    default_params={
        "top_k": 5,
    },
    capabilities=frozenset(
        name
        for name in dir(TopKSamplingStrategy)
        if callable(getattr(TopKSamplingStrategy, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=TopKSamplingStrategy,
    schema_contribution=None,
)

QASpanTopKDecodingStrategy_aliases = []
for alias, canonical_key in DECODING_LOOKUP_MAP.items():
    if canonical_key == "qaspantopkdecoding":
        QASpanTopKDecodingStrategy_aliases.append(alias) if alias != "qaspantopkdecoding" else None
QASpanTopKDecodingStrategy_aliases = tuple(QASpanTopKDecodingStrategy_aliases)

QASpanTopKDecodingStrategy_PARAMS_SCHEMA = MappingSchema(
    fields={
        "max_answer_length": ScalarSchema(
            python_types=(int,),
            default=30,
        ),
        "n_best_size": ScalarSchema(
            python_types=(int,),
            default=20,
        ),
    },
    allow_unknown=False,
    allow_none=True,
)

QASpanTopKDecodingStrategy_SPEC = ComponentSpec(
    category="decoding",
    key="qaspantopkdecoding",
    aliases=QASpanTopKDecodingStrategy_aliases,
    params_schema=QASpanTopKDecodingStrategy_PARAMS_SCHEMA,
    default_params={
        "max_answer_length": 30,
        "n_best_size": 20,
    },
    capabilities=frozenset(
        name
        for name in dir(QASpanTopKDecodingStrategy)
        if callable(getattr(QASpanTopKDecodingStrategy, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=QASpanTopKDecodingStrategy,
    schema_contribution=None,
)

# TASK
TASK_CANONICAL_MAP = TASK_BUILDER_REGISTRY.canonical_map
TASK_LOOKUP_MAP = TASK_BUILDER_REGISTRY.lookup_map

BIULD_MLM_PRETRAIN = TASK_CANONICAL_MAP["mlmpretrain_train"]
BUILD_MLM_TEST = TASK_CANONICAL_MAP["mlmpretrain_test"]
BUILD_MLM_INFERENCE = TASK_CANONICAL_MAP["mlmpretrain_inference"]
# AI-CHANGE: MLM 可视化任务 Builder(AI 编写, 待人工审查)。
# 只组装 tokenizer/model/dataloader/visualizers,不组装损失与指标。
BUILD_MLM_VISUALIZE = TASK_CANONICAL_MAP["mlmpretrain_visualize"]
BUILD_QA_SFT = TASK_CANONICAL_MAP["qasft_train"]
BUILD_QA_TEST = TASK_CANONICAL_MAP["qasft_test"]
BUILD_QA_INFERENCE = TASK_CANONICAL_MAP["qasft_inference"]
BUILD_QA_VISUALIZE = TASK_CANONICAL_MAP["qasft_visualize"]

BIULD_MLM_PRETRAIN_aliases = []
BUILD_MLM_TEST_aliases = []
BUILD_MLM_INFERENCE_aliases = []
# AI-CHANGE: MLM 可视化任务别名(AI 编写, 待人工审查)。
BUILD_MLM_VISUALIZE_aliases = []
BUILD_QA_SFT_aliases = []
BUILD_QA_TEST_aliases = []
BUILD_QA_INFERENCE_aliases = []
BUILD_QA_VISUALIZE_aliases = []

for alias, canonical_key in TASK_LOOKUP_MAP.items():
    if canonical_key == "mlmpretrain_train":
        BIULD_MLM_PRETRAIN_aliases.append(alias) if alias != "mlmpretrain_train" else None
    elif canonical_key == "mlmpretrain_test":
        BUILD_MLM_TEST_aliases.append(alias) if alias != "mlmpretrain_test" else None
    elif canonical_key == "mlmpretrain_inference":
        BUILD_MLM_INFERENCE_aliases.append(alias) if alias != "mlmpretrain_inference" else None
    elif canonical_key == "mlmpretrain_visualize":
        BUILD_MLM_VISUALIZE_aliases.append(alias) if alias != "mlmpretrain_visualize" else None
    elif canonical_key == "qasft_train":
        BUILD_QA_SFT_aliases.append(alias) if alias != "qasft_train" else None
    elif canonical_key == "qasft_test":
        BUILD_QA_TEST_aliases.append(alias) if alias != "qasft_test" else None
    elif canonical_key == "qasft_inference":
        BUILD_QA_INFERENCE_aliases.append(alias) if alias != "qasft_inference" else None
    elif canonical_key == "qasft_visualize":
        BUILD_QA_VISUALIZE_aliases.append(alias) if alias != "qasft_visualize" else None

BIULD_MLM_PRETRAIN_aliases = tuple(BIULD_MLM_PRETRAIN_aliases)
BUILD_MLM_TEST_aliases = tuple(BUILD_MLM_TEST_aliases)
BUILD_MLM_INFERENCE_aliases = tuple(BUILD_MLM_INFERENCE_aliases)
# AI-CHANGE: MLM 可视化任务别名(AI 编写, 待人工审查)。
BUILD_MLM_VISUALIZE_aliases = tuple(BUILD_MLM_VISUALIZE_aliases)
BUILD_QA_SFT_aliases = tuple(BUILD_QA_SFT_aliases)
BUILD_QA_TEST_aliases = tuple(BUILD_QA_TEST_aliases)
BUILD_QA_INFERENCE_aliases = tuple(BUILD_QA_INFERENCE_aliases)
BUILD_QA_VISUALIZE_aliases = tuple(BUILD_QA_VISUALIZE_aliases)

MLM_PRETRAIN_TRAIN_BOOTSTRAP_DESCRIPTORS = (
    BootstrapDescriptor(("tokenizer",), "tokenizer"),
    BootstrapDescriptor(("model",), "model"),
    BootstrapDescriptor(("losses", "*"), "loss"),
    BootstrapDescriptor(("optimization", "optimizer"), "optimizer"),
    BootstrapDescriptor(("optimization", "scheduler"), "scheduler"),
    BootstrapDescriptor(("evaluation", "metrics", "train", "*"), "metric"),
    BootstrapDescriptor(("evaluation", "metrics", "valid", "*"), "metric"),
    BootstrapDescriptor(("evaluation", "metrics", "test", "*"), "metric"),
)

MLM_PRETRAIN_TEST_BOOTSTRAP_DESCRIPTORS = (
    # PARSE_REFACTOR_DEMO: Test YAML 可以声明 losses，因此 Bootstrap 必须
    # 为每个 Loss 建立 ComponentSelection，供 PolymorphicSchema 使用。
    BootstrapDescriptor(("losses", "*"), "loss", required=False),
    BootstrapDescriptor(("evaluation", "metrics", "test", "*"), "metric"),
)

MLM_PRETRAIN_INFERENCE_BOOTSTRAP_DESCRIPTORS = (
    BootstrapDescriptor(("inference", "decoding"), "decoding"),
)

QA_SFT_TRAIN_BOOTSTRAP_DESCRIPTORS = (
    BootstrapDescriptor(("tokenizer",), "tokenizer"),
    BootstrapDescriptor(("model",), "model"),
    BootstrapDescriptor(("losses", "*"), "loss"),
    BootstrapDescriptor(("optimization", "optimizer"), "optimizer"),
    BootstrapDescriptor(("optimization", "scheduler"), "scheduler"),
    BootstrapDescriptor(("evaluation", "metrics", "train", "*"), "metric"),
    BootstrapDescriptor(("evaluation", "metrics", "valid", "*"), "metric"),
    BootstrapDescriptor(("evaluation", "metrics", "test", "*"), "metric"),
)

QA_SFT_TEST_BOOTSTRAP_DESCRIPTORS = (
    # PARSE_REFACTOR_DEMO: 与 MLM Test 一样，QA Test 的 Loss 可以由 Test
    # 配置声明；未声明时由 checkpoint.train_config 提供。
    BootstrapDescriptor(("losses", "*"), "loss", required=False),
    BootstrapDescriptor(("evaluation", "metrics", "test", "*"), "metric"),
)

QA_SFT_INFERENCE_BOOTSTRAP_DESCRIPTORS = (
    BootstrapDescriptor(("inference", "decoding"), "decoding"),
)

QA_SFT_VISUALIZE_BOOTSTRAP_DESCRIPTORS = (
    BootstrapDescriptor(("visualize", "methods", "*"), "visualizer"),
    BootstrapDescriptor(
        ("visualize", "methods", "*", "params", "losses", "*"),
        "loss",
        required=False,
    ),
)

BIULD_MLM_PRETRAIN_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_MLM_TEST_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_MLM_INFERENCE_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_QA_SFT_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_QA_TEST_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_QA_INFERENCE_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

# AI-CHANGE: MLM 可视化任务 Params Schema(AI 编写, 待人工审查)。
# 任务级 params 为空占位,方法级 params 由各 visualizer Spec 负责。
BUILD_MLM_VISUALIZE_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_MLM_PRETRAIN_SPEC = ComponentSpec(
    category="task",
    key="mlmpretrain_train",
    aliases=BIULD_MLM_PRETRAIN_aliases,
    params_schema=BIULD_MLM_PRETRAIN_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(BIULD_MLM_PRETRAIN)
        if callable(getattr(BIULD_MLM_PRETRAIN, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BIULD_MLM_PRETRAIN,
    schema_contribution=None,
    bootstrap_descriptors=MLM_PRETRAIN_TRAIN_BOOTSTRAP_DESCRIPTORS,
)

BUILD_MLM_TEST_SPEC = ComponentSpec(
    category="task",
    key="mlmpretrain_test",
    aliases=BUILD_MLM_TEST_aliases,
    params_schema=BUILD_MLM_TEST_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(BUILD_MLM_TEST)
        if callable(getattr(BUILD_MLM_TEST, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_MLM_TEST,
    schema_contribution=None,
    bootstrap_descriptors=MLM_PRETRAIN_TEST_BOOTSTRAP_DESCRIPTORS,
)

BUILD_MLM_INFERENCE_SPEC = ComponentSpec(
    category="task",
    key="mlmpretrain_inference",
    aliases=BUILD_MLM_INFERENCE_aliases,
    params_schema=BUILD_MLM_INFERENCE_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(BUILD_MLM_INFERENCE)
        if callable(getattr(BUILD_MLM_INFERENCE, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_MLM_INFERENCE,
    schema_contribution=None,
    bootstrap_descriptors=MLM_PRETRAIN_INFERENCE_BOOTSTRAP_DESCRIPTORS,
)

BUILD_QA_INFERENCE_SPEC = ComponentSpec(
    category="task",
    key="qasft_inference",
    aliases=BUILD_QA_INFERENCE_aliases,
    params_schema=BUILD_QA_INFERENCE_PARAMS_SCHEMA,
    default_params={},
    capabilities=frozenset(
        name
        for name in dir(BUILD_QA_INFERENCE)
        if callable(getattr(BUILD_QA_INFERENCE, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_QA_INFERENCE,
    schema_contribution=None,
    bootstrap_descriptors=QA_SFT_INFERENCE_BOOTSTRAP_DESCRIPTORS,
)

BUILD_QA_SFT_SPEC = ComponentSpec(
    category="task",
    key="qasft_train",
    aliases=BUILD_QA_SFT_aliases,
    params_schema=BUILD_QA_SFT_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(BUILD_QA_SFT)
        if callable(getattr(BUILD_QA_SFT, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_QA_SFT,
    schema_contribution=None,
    bootstrap_descriptors=QA_SFT_TRAIN_BOOTSTRAP_DESCRIPTORS,
)

BUILD_QA_VISUALIZE_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_QA_VISUALIZE_SPEC = ComponentSpec(
    category="task",
    key="qasft_visualize",
    aliases=BUILD_QA_VISUALIZE_aliases,
    params_schema=BUILD_QA_VISUALIZE_PARAMS_SCHEMA,
    default_params={},
    capabilities=frozenset(
        name
        for name in dir(BUILD_QA_VISUALIZE)
        if callable(getattr(BUILD_QA_VISUALIZE, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_QA_VISUALIZE,
    schema_contribution=None,
    bootstrap_descriptors=QA_SFT_VISUALIZE_BOOTSTRAP_DESCRIPTORS,
)

BUILD_QA_TEST_SPEC = ComponentSpec(
    category="task",
    key="qasft_test",
    aliases=BUILD_QA_TEST_aliases,
    params_schema=BUILD_QA_TEST_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(BUILD_QA_TEST)
        if callable(getattr(BUILD_QA_TEST, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_QA_TEST,
    schema_contribution=None,
    bootstrap_descriptors=QA_SFT_TEST_BOOTSTRAP_DESCRIPTORS,
)

BUILD_QA_INFERENCE_PARAMS_SCHEMA = MappingSchema(
    fields={},
    allow_unknown=True,
    allow_none=False,
)

BUILD_QA_INFERENCE_SPEC = ComponentSpec(
    category="task",
    key="qasft_inference",
    aliases=BUILD_QA_INFERENCE_aliases,
    params_schema=BUILD_QA_INFERENCE_PARAMS_SCHEMA,
    default_params={},
    capabilities=frozenset(
        name
        for name in dir(BUILD_QA_INFERENCE)
        if callable(getattr(BUILD_QA_INFERENCE, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_QA_INFERENCE,
    schema_contribution=None,
    bootstrap_descriptors=QA_SFT_INFERENCE_BOOTSTRAP_DESCRIPTORS,
)

# AI-CHANGE: MLM 可视化任务 Bootstrap 描述符(AI 编写, 待人工审查)。
# 只扫描 visualize.methods.* 方法节点,不扫描 tokenizer/model/losses；
# 模型结构从 checkpoint.train_config 恢复,数据集走 data/dataset 静态声明。
MLM_PRETRAIN_VISUALIZE_BOOTSTRAP_DESCRIPTORS = (
    BootstrapDescriptor(("visualize", "methods", "*"), "visualizer"),
    BootstrapDescriptor(
        ("visualize", "methods", "*", "params", "losses", "*"),
        "loss",
        required=False,
    ),
)

# AI-CHANGE: MLM 可视化任务 Spec(AI 编写, 待人工审查)。
BUILD_MLM_VISUALIZE_SPEC = ComponentSpec(
    category="task",
    key="mlmpretrain_visualize",
    aliases=BUILD_MLM_VISUALIZE_aliases,
    params_schema=BUILD_MLM_VISUALIZE_PARAMS_SCHEMA,
    default_params={

    },
    capabilities=frozenset(
        name
        for name in dir(BUILD_MLM_VISUALIZE)
        if callable(getattr(BUILD_MLM_VISUALIZE, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=BUILD_MLM_VISUALIZE,
    schema_contribution=None,
    bootstrap_descriptors=MLM_PRETRAIN_VISUALIZE_BOOTSTRAP_DESCRIPTORS,
)

# VISUALIZE
# AI-CHANGE: 可视化方法 Spec(AI 编写, 待人工审查)。
# 只覆盖方法级组件(layer/head/max_samples),不包含模型、数据集与任务 Builder
# 的声明；任务级 Spec 待模型与数据集契约确定后再补充。
VISUALIZER_CANONICAL_MAP = VISUALIZER_REGISTRY.canonical_map
VISUALIZER_LOOKUP_MAP = VISUALIZER_REGISTRY.lookup_map

AttentionHeatmapVisualizer = VISUALIZER_CANONICAL_MAP["attentionheatmap"]

AttentionHeatmapVisualizer_aliases = []
for alias, canonical_key in VISUALIZER_LOOKUP_MAP.items():
    if canonical_key == "attentionheatmap":
        AttentionHeatmapVisualizer_aliases.append(alias) if alias != "attentionheatmap" else None
AttentionHeatmapVisualizer_aliases = tuple(AttentionHeatmapVisualizer_aliases)

# ATTENTION_HEATMAP_PARAMS_SCHEMA
ATTENTION_HEATMAP_PARAMS_SCHEMA = MappingSchema(
    fields={
        "layer": ScalarSchema(
            python_types=(int, str, list),
            default=-1,
        ),
        "head": ScalarSchema(
            python_types=(int, str, list),
            default=0,
        ),
        "max_samples": ScalarSchema(
            python_types=(int,),
            default=5,
        ),
    },
    allow_unknown=False,
    allow_none=False,
)

LOSS_LANDSCAPE_PARAMS_SCHEMA = MappingSchema(
    fields={
        "normalization": EnumSchema(
            choices={
                "per_parameter": "per_parameter",
                "global": "global",
            },
            default="per_parameter",
        ),
        "output_format": EnumSchema(
            choices={
                "png": "png",
                "html": "html",
                "both": "both",
            },
            default="both",
        ),
        "grid_size": ScalarSchema(python_types=(int,), default=11),
        "alpha_min": ScalarSchema(python_types=(int, float), default=-1.0),
        "alpha_max": ScalarSchema(python_types=(int, float), default=1.0),
        "beta_min": ScalarSchema(python_types=(int, float), default=-1.0),
        "beta_max": ScalarSchema(python_types=(int, float), default=1.0),
        "batch_count": ScalarSchema(python_types=(int,), default=1),
        "losses": SequenceSchema(
            item_schema=PolymorphicSchema(
                category="loss",
                common_schema=MappingSchema(
                    fields={
                        "name": ScalarSchema(python_types=(str,), required=True),
                        "type": RegistryKeySchema(registry=LOSS_REGISTRY),
                        "weight": ScalarSchema(python_types=(int, float), default=1.0),
                        "params": MappingSchema(
                            fields={},
                            allow_unknown=True,
                            allow_none=True,
                        ),
                    },
                    allow_unknown=False,
                    allow_none=False,
                ),
            ),
            min_items=1,
            required=True,
        ),
    },
    allow_unknown=False,
    allow_none=False,
)

LossLandscapeVisualizer_aliases = []
for alias, canonical_key in VISUALIZER_LOOKUP_MAP.items():
    if canonical_key == "losslandscape":
        LossLandscapeVisualizer_aliases.append(alias) if alias != "losslandscape" else None
LossLandscapeVisualizer_aliases = tuple(LossLandscapeVisualizer_aliases)

LOSS_LANDSCAPE_SPEC = ComponentSpec(
    category="visualizer",
    key="losslandscape",
    aliases=LossLandscapeVisualizer_aliases,
    params_schema=LOSS_LANDSCAPE_PARAMS_SCHEMA,
    default_params={
        "normalization": "per_parameter",
        "output_format": "both",
        "grid_size": 11,
        "alpha_min": -1.0,
        "alpha_max": 1.0,
        "beta_min": -1.0,
        "beta_max": 1.0,
        "batch_count": 1,
    },
    capabilities=frozenset(
        name
        for name in dir(LossLandscapeVisualizer)
        if callable(getattr(LossLandscapeVisualizer, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=LossLandscapeVisualizer,
    schema_contribution=None,
)

ATTENTION_HEATMAP_SPEC = ComponentSpec(
    category="visualizer",
    key="attentionheatmap",
    aliases=AttentionHeatmapVisualizer_aliases,
    params_schema=ATTENTION_HEATMAP_PARAMS_SCHEMA,
    default_params={
        "layer": -1,
        "head": 0,
        "max_samples": 5,
    },
    capabilities=frozenset(
        name
        for name in dir(AttentionHeatmapVisualizer)
        if callable(getattr(AttentionHeatmapVisualizer, name))
        and not name.startswith("_")
    ),
    normalizers=(),
    validators=(),
    factory=AttentionHeatmapVisualizer,
    schema_contribution=None,
)