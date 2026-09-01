from collections.abc import Mapping, Callable
from dataclasses import dataclass
from typing import Any
from copy import deepcopy

from parse.errors import UnknownFieldError, TypeMismatchError
from parse.schema.nodes import MappingSchema, SchemaNode
from parse.specs.validator import ValidatorSpec
from parse.schema.merge import merge_schema


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

