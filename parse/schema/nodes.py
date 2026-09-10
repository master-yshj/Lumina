import pathlib

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal
from copy import deepcopy
from collections.abc import Mapping

from parse.context import MISSING
from parse.errors import ConfigError, MissingFieldError, NullNotAllowedError, TypeMismatchError, \
    ParameterNormalizationError, UnknownFieldError, SchemaError, UnknownRegistryKeyError, BootstrapError, BootstrapError


@dataclass(frozen=True)
class SchemaNode(ABC):
    """
    SchemaNode 抽象Schema节点基类
    """

    @abstractmethod
    def normalize(self, value, path, context) -> Any:
        raise NotImplementedError

@dataclass(frozen=True)
class ScalarSchema(SchemaNode):
    """
    ScalarSchema 普通叶子节点类
    """
    python_types: tuple[type, ...]
    required: bool = False
    allow_none: bool = False
    default: Any = MISSING
    normalizer: Callable | None = None

    def __post_init__(self):

        if not isinstance(self.python_types, tuple):
            raise TypeError(f"python_types: {self.python_types} 必须为tuple类型的喵! 现在的type: {type(self.python_types)}")
        if not self.python_types:
            raise ValueError(f"python_types 不能为空喵!")
        if not all(isinstance(python_type, type) for python_type in self.python_types):
            raise TypeError(f"python_types 中的每一个元素都必须为python类型对象喵!")

        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型的喵! 现在的type: {type(self.required)}")

        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 现在的type: {type(self.allow_none)}")

        if self.required is True and self.default is not MISSING:
            raise ValueError(f"required 为True时，default必须为MISSING类型的喵! 现在default的type: {type(self.default)}")

        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError(f"normalizer 必须可以被调用喵!")

    def normalize(self, value, path, context) -> Any:
        if value is MISSING:
            if self.default is not MISSING:
                return deepcopy(self.default)
            if self.required:
                raise MissingFieldError(f"path: {path} 配置中缺少了该字段喵!")
            else:
                return MISSING
        elif value is None:
            if self.allow_none:
                return None
            else:
                raise NullNotAllowedError(f"path: {path}: {value} 该配置的值不能为空喵!")
        else:
            if not isinstance(value, self.python_types):
                raise TypeMismatchError(f"path: {path}: {value} 不属于python类型喵! 允许的type: {self.python_types}")
            if self.normalizer is None:
                return value
            else:
                try:
                    return self.normalizer(value)
                except ConfigError:
                    raise
                except (ValueError, TypeError) as e:
                    raise ParameterNormalizationError(f"path: {path}: {value} 规范化时出现错误了喵!") from e

@dataclass(frozen=True)
class MappingSchema(SchemaNode):
    """
    MappingSchema 固定字典节点类
    """
    fields: Mapping[str, SchemaNode]
    allow_unknown: bool = False
    allow_none: bool = False

    def __post_init__(self):
        if not isinstance(self.fields, Mapping):
            raise TypeError(f"fields 必须为Mapping类型的喵! 现在的type: {type(self.fields)}")
        for key, value in self.fields.items():
            if not isinstance(key, str):
                raise TypeError(f"fields: {key} 这个键必须为str类型的喵! 现在的type: {type(key)}")
            if not key.strip():
                raise ValueError(f"fields 中的键不能为空喵!")
            if not isinstance(value, SchemaNode):
                raise TypeError(f"fields.{key}: {value} 必须为SchemaNode类型的喵! 现在的type: {type(value)}")
        if not type(self.allow_unknown) is bool:
            raise TypeError(f"allow_unknown: {self.allow_unknown} 必须为bool类型的喵! 现在的type: {type(self.allow_unknown)}")
        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 现在的type: {type(self.allow_none)}")

    def normalize(self, value, path, context) -> Any:
        if value is MISSING:
            return MISSING
        if value is None:
            if self.allow_none:
                return None
            else:
                raise NullNotAllowedError(f"path: {path}: {value} 该配置的值不能为空喵!")
        if not isinstance(value, Mapping):
            raise TypeMismatchError(f"path: {path}: {value} 必须为Mapping类型的喵! 现在的type: {type(value)}")
        unknown_keys = value.keys() - self.fields.keys()
        if unknown_keys and self.allow_unknown is False:
            raise UnknownFieldError(f"path: {path} 中出现了未知字段喵! unknown_keys: {unknown_keys}")
        result = {}
        for field_name, field_schema in self.fields.items():
            child_path = path.child(field_name)
            if field_name in value:
                raw_value = value[field_name]
            else:
                raw_value = MISSING
            normalized_value = field_schema.normalize(
                raw_value,
                child_path,
                context,
            )
            if normalized_value is MISSING:
                pass
            else:
                result[field_name] = normalized_value
        if self.allow_unknown:
            for unknown_key in unknown_keys:
                result[unknown_key] = deepcopy(value[unknown_key])
        return result

@dataclass(frozen=True)
class SequenceSchema(SchemaNode):
    """
    SequenceSchema 固定列表节点类
    """
    item_schema: SchemaNode
    min_items: int = 0
    max_items: int | None = None
    allow_none: bool = False
    default: Any = MISSING
    required: bool = False

    def __post_init__(self):
        if not isinstance(self.item_schema, SchemaNode):
            raise TypeError(f"item_schema 必须为SchemaNode类型的喵! 现在的type: {type(self.item_schema)}")
        if not type(self.min_items) is int:
            raise TypeError(f"min_items: {self.min_items} 必须为严格int类型的喵! 现在的type: {type(self.min_items)}")
        if self.min_items < 0:
            raise ValueError(f"min_items: {self.min_items} 必须为非负整数喵!")
        if self.max_items is not None and type(self.max_items) is not int:
            raise TypeError(f"max_items: {self.max_items} 必须为int类型或者None喵! 现在的type: {type(self.max_items)}")
        if self.max_items is not None and self.max_items < 0:
            raise ValueError(f"max_items: {self.max_items} 不为空时必须为正整数喵!")
        if self.max_items is not None and self.max_items < self.min_items:
            raise ValueError(f"max_items: {self.max_items} min_items: {self.min_items} min_items 不能大于或等于 max_items 喵!")
        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 现在的type: {type(self.allow_none)}")
        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型的喵! 现在的type: {type(self.required)}")
        if self.required is True and self.default is not MISSING:
            raise ValueError(f"required 为True时，default必须为MISSING类型的喵! 现在default的type: {type(self.default)}")
        if self.default is not MISSING and not isinstance(self.default, list):
            raise TypeError(f"default: {self.default} 不为MISSING时必须为list类型的喵! 现在的type: {type(self.default)}")
        if self.default is not MISSING:
            default_length = len(self.default)
            if (self.max_items is not None and default_length > self.max_items) or default_length < self.min_items:
                raise ValueError(f"default的长度: {default_length} 不合法喵! 合法的范围: [{self.min_items}, {self.max_items}]")


    def normalize(self, value, path, context) -> Any:
        if value is MISSING:
            if self.default is not MISSING:
                return deepcopy(self.default)
            if self.required:
                raise MissingFieldError(f"path: {path} 配置中缺少了该字段喵!")
            else:
                return MISSING
        elif value is None:
            if self.allow_none is True:
                return None
            else:
                raise NullNotAllowedError(f"path: {path}: {value} 该配置的值不能为空喵!")
        else:
            if not isinstance(value, list):
                raise TypeMismatchError(f"path: {path}: {value} 不属于list类型喵! 现在的type: {type(value)}")
            value_length = len(value)
            if (self.max_items is not None and value_length > self.max_items ) or value_length < self.min_items:
                raise SchemaError(f"path: {path}: {value} 该值的长度为 {value_length} 不在范围内喵! 合法的范围: [{self.min_items},{self.max_items}]")
            result_list = []
            for idx, single_value in enumerate(value):
                normalized_value = self.item_schema.normalize(
                    single_value,
                    path.index(idx),
                    context,
                )
                if normalized_value is MISSING:
                    pass
                else:
                    result_list.append(normalized_value)
            return result_list

@dataclass(frozen=True)
class EnumSchema(SchemaNode):
    """
    EnumSchema 有限字符串节点类
    """
    choices: Mapping[str, str] | set[str]
    case_sensitive: bool = False
    required: bool = False
    allow_none: bool = False
    default: Any = MISSING

    def __post_init__(self):
        if not isinstance(self.choices, (Mapping,set)):
            raise TypeError(f"choices: {self.choices} 必须为Mapping或者set类型的喵! 现在的type: {type(self.choices)}")
        if not type(self.case_sensitive) is bool:
            raise TypeError(f"case_sensitive: {self.case_sensitive} 必须为bool类型的喵! 当前的type: {type(self.case_sensitive)}")
        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型的喵! 当前的type: {type(self.required)}")
        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 当前的type: {type(self.allow_none)}")
        if self.required is True and self.default is not MISSING:
            raise ValueError(f"required 为True时，default必须为MISSING类型的喵! 现在default的type: {type(self.default)}")
        if not self.choices:
            raise ValueError(f"choices 不能为空喵!")
        if isinstance(self.choices, Mapping):
            seen_lookup_keys = set()
            for choice_key, choice_value in self.choices.items():
                if not isinstance(choice_key, str):
                    raise TypeError(f"choices 中的键必须为str类型的喵! 当前key: {choice_key}, type: {type(choice_key)}")
                if not isinstance(choice_value, str):
                    raise TypeError(f"choices 中的值必须为str类型的喵! 当前value: {choice_value}, type: {type(choice_value)}")
                if not choice_key.strip():
                    raise ValueError(f"choices 中的键不能为空喵!")
                if not choice_value.strip():
                    raise ValueError(f"choices 中的值不能为空喵!")
                if self.case_sensitive:
                    choice_lookup_key = choice_key
                else:
                    choice_lookup_key = choice_key.strip().casefold()
                if choice_lookup_key in seen_lookup_keys:
                    raise ValueError(f"choice 中不能出现重复(不区分大小写)的key: {choice_key} 喵! ")
                seen_lookup_keys.add(choice_lookup_key)
        if isinstance(self.choices, set):
            if self.default is not MISSING:
                if self.default not in self.choices:
                    raise ValueError(f"default: {self.default} 的值不合法喵! 合法的值有: {self.choices}")
            seen_lookup_keys = set()
            for choice in self.choices:
                if not isinstance(choice, str):
                    raise TypeError(f"choices 中的元素必须为str类型的喵!")
                if not choice.strip():
                    raise ValueError(f"choices 中的元素不能为空喵!")
                if self.case_sensitive:
                    choice_lookup_key = choice
                else:
                    choice_lookup_key = choice.strip().casefold()
                if choice_lookup_key in seen_lookup_keys:
                    raise ValueError(f"choice 中不能出现重复(不区分大小写)的元素: {choice} 喵! ")
                seen_lookup_keys.add(choice_lookup_key)
        if self.default is not MISSING:
            if not isinstance(self.default, str):
                raise TypeError(f"default 不为MISSING时必须为str类型的喵! 现在的type: {type(self.default)}")
            if isinstance(self.choices, Mapping):
                canonical_values = set(self.choices.values())
            else:
                canonical_values = set(self.choices)
            if self.default not in canonical_values:
                raise ValueError(f"default: {self.default} 必须是canonical value喵! 合法值: {canonical_values}")

    def normalize(self, value, path, context) -> Any:
        if value is MISSING:
            if self.default is not MISSING:
                return deepcopy(self.default)
            if self.required:
                raise MissingFieldError(f"path: {path} 该配置中缺失了该字段喵!")
            else:
                return MISSING
        elif value is None:
            if self.allow_none is True:
                return None
            else:
                raise NullNotAllowedError(f"path: {path}: {value} 该配置的值不能为空喵!")
        else:
            if not isinstance(value, str):
                raise TypeMismatchError(f"path: {path}: {value} 必须为字符串类型的喵! 现在的type: {type(value)}")
            if self.case_sensitive:
                lookup_key = value
            else:
                lookup_key = value.strip().casefold()
            if isinstance(self.choices, Mapping):
                for choice_key, choice_value in self.choices.items():
                    if self.case_sensitive:
                        choice_lookup_key = choice_key
                    else:
                        choice_lookup_key = choice_key.strip().casefold()
                    if choice_lookup_key == lookup_key:
                        return choice_value
            if isinstance(self.choices, set):
                for choice in self.choices:
                    if self.case_sensitive:
                        choice_lookup_key = choice
                    else:
                        choice_lookup_key = choice.strip().casefold()
                    if choice_lookup_key == lookup_key:
                        return choice
            raise SchemaError(f"path: {path}: {value} 不合法喵! 允许: {self.choices}")



@dataclass(frozen=True)
class RegistryKeySchema(SchemaNode):
    """
    Registry键节点类
    """
    registry: Any
    required: bool = True
    allow_none: bool = False

    def __post_init__(self):
        if self.registry is None:
            raise TypeError(f"registry: {self.registry} 不能为空喵!")
        if not hasattr(self.registry, "resolve_key") or not callable(self.registry.resolve_key):
            raise TypeError(f"registry: {self.registry} 中没有可以调用的resolve_key喵!")
        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型的喵! 现在的type: {type(self.required)}")
        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 现在的type: {type(self.allow_none)}")

    def normalize(self, value, path, context) -> Any:
        if value is MISSING:
            if self.required:
                raise MissingFieldError(f"path: {path} 该配置中缺失了该字段喵!")
            else:
                return MISSING
        elif value is None:
            if self.allow_none is True:
                return None
            else:
                raise NullNotAllowedError(f"path: {path}: {value} 该配置的值不能为空喵!")
        else:
            if not isinstance(value, str):
                raise TypeMismatchError(f"path: {path}: {value} 必须为str类型的喵! 现在的type: {type(value)}")
            if not value.strip():
                raise SchemaError(f"path: {path}: {value} 注册表的键不能为空喵!")
            try:
                resolved_value = self.registry.resolve_key(value)
            except UnknownRegistryKeyError:
                raise
            except Exception:
                raise
            return resolved_value

@dataclass(frozen=True)
class PolymorphicSchema(SchemaNode):
    """
    PolymorphicSchema: 多态组件外壳类
    """
    category: str
    common_schema: MappingSchema
    discriminator: str = "type"
    params_field: str = "params"

    def __post_init__(self):
        if not isinstance(self.category, str):
            raise TypeError(f"category: {self.category} 必须为str类型的喵! 现在的type: {type(self.category)}")
        if not self.category.strip():
            raise ValueError(f"category 不能为空喵!")
        if not isinstance(self.discriminator, str):
            raise TypeError(f"discriminator: {self.discriminator} 必须为str类型的喵! 现在的type: {type(self.discriminator)}")
        if not self.discriminator.strip():
            raise ValueError(f"discriminator 不能为空喵!")
        if not isinstance(self.params_field, str):
            raise TypeError(f"params_field: {self.params_field} 必须为str类型的喵! 现在的type: {type(self.params_field)}")
        if not self.params_field.strip():
            raise ValueError(f"params_field 不能为空喵!")
        if self.discriminator == self.params_field:
            raise ValueError(f"discriminator: {self.discriminator}\n"
                             f"params_field: {self.params_field}\n"
                             f"不能指向同一字段喵!")
        if not isinstance(self.common_schema, MappingSchema):
            raise TypeError(f"common_schema: {self.common_schema} 必须为MappingSchema喵! 现在的type: {type(self.common_schema)}")
        if not self.discriminator in self.common_schema.fields:
            raise ValueError(f"discriminator: {self.discriminator} 必须在 common_schema.fields: {self.common_schema.fields} 中喵!")
        if not self.params_field in self.common_schema.fields:
            raise ValueError(f"params_field: {self.params_field} 必须在 common_schema.fields: {self.common_schema.fields} 中喵!")

    def normalize(self, value, path, context) -> Any:
        if not isinstance(value, Mapping):
            raise TypeMismatchError(f"path: {path}: {value} 必须为Mapping喵! 现在的type: {type(value)}")
        try:
            normalized_value = self.common_schema.normalize(value, path, context)
        except (MissingFieldError, UnknownFieldError, TypeMismatchError, UnknownRegistryKeyError):
            raise
        try:
            normalized_discriminator = normalized_value[self.discriminator]
        except KeyError as e:
            raise SchemaError(f"common_schema 不符合 PolymorphicSchema 契约喵! 找不到 discriminator 喵!")
        selection = context.get_selection(path)
        if selection.path != path:
            raise BootstrapError(f"selection_path: {selection.path} 与当前path: {path} 不一致喵!")
        if selection.category != self.category:
            raise BootstrapError(f"selection_category: {selection.category} 与当前category: {self.category} 不一致喵!")
        if selection.key != normalized_discriminator:
            raise BootstrapError(f"selection_key: {selection.key} 与当前normalized_discriminator: {normalized_discriminator} 不一致喵!")
        if selection.spec is None:
            raise BootstrapError(f"selection_spec 不能为空喵!")
        if not hasattr(selection.spec, "normalize_params"):
            raise BootstrapError(f"selection_spec 必须提供normalize_params方法喵!")
        if not hasattr(selection.spec, "key") or selection.spec.key != normalized_discriminator:
            raise BootstrapError(
                f"selection_spec.key: {getattr(selection.spec, 'key', None)} "
                f"与normalized_discriminator: {normalized_discriminator} 不一致喵!"
            )
        try:
            normalized_params = normalized_value[self.params_field]
        except KeyError:
            normalized_params = MISSING
        if normalized_params is not MISSING and normalized_params is not None:
            if not isinstance(normalized_params, Mapping):
                raise TypeMismatchError(
                    f"path: {path.child(self.params_field)}: {normalized_params} "
                    f"必须为None或者Mapping类型的喵! "
                    f"现在的type: {type(normalized_params)}"
                )
        raw_params = None if normalized_params is MISSING else normalized_params
        normalized_params = selection.spec.normalize_params(
            raw_params=raw_params,
            path=path.child(self.params_field),
            context=context,
        )
        if not isinstance(normalized_params, Mapping):
            raise SchemaError(
                f"path: {path.child(self.params_field)} 的ComponentSpec.normalize_params() "
                f"必须返回Mapping类型喵! 现在的type: {type(normalized_params)}"
            )
        result = deepcopy(normalized_value)
        result[self.params_field] = deepcopy(normalized_params)
        return result

@dataclass(frozen=True)
class PathSchema(SchemaNode):
    """
    PathSchema 路径描述节点类
    """
    base: Any
    kind: Literal["file", "directory"]
    required: bool = False
    default: Any = MISSING
    allow_none: bool = False
    allowed_suffixes: tuple[str, ...] = ()
    must_exist: bool = False
    non_empty: bool = False
    resolver_name: str | None = None

    def __post_init__(self):
        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型的喵! 现在的type: {type(self.required)}")
        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 现在的type: {type(self.allow_none)}")
        if not type(self.must_exist) is bool:
            raise TypeError(f"must_exist: {self.must_exist} 必须为bool类型的喵! 现在的type: {type(self.must_exist)}")
        if not type(self.non_empty) is bool:
            raise TypeError(f"non_empty: {self.non_empty} 必须为bool类型的喵! 现在的type: {type(self.non_empty)}")
        if self.required is True and self.default is not MISSING:
            raise ValueError(f"在 required: {self.required} 时, default: {self.default} 必须为MISSING喵!")
        if self.default is not MISSING and not isinstance(self.default, pathlib.Path):
            raise TypeError(f"default 不为MISSING时必须为pathlib.Path类型喵! 现在的type: {type(self.default)}")
        if not isinstance(self.kind, str):
            raise TypeError(f"kind: {self.kind} 必须为str类型的喵! 现在的type: {type(self.kind)}")
        if not self.kind in ["file", "directory"]:
            raise ValueError(f"kind: {self.kind} 必须属于 ['file','directory'] 喵!")
        if not isinstance(self.allowed_suffixes, tuple):
            raise TypeError(
                f"allowed_suffixes: {self.allowed_suffixes} 必须为tuple类型的喵! "
                f"现在的type: {type(self.allowed_suffixes)}"
            )

        for suffix in self.allowed_suffixes:
            if not isinstance(suffix, str):
                raise TypeError(
                    f"allowed_suffixes 中的元素必须为str类型的喵! "
                    f"当前元素: {suffix}"
                )
            if not suffix.strip():
                raise ValueError(
                    f"allowed_suffixes 中每个后缀不能为空喵!"
                    f"当前元素: {suffix}"
                )
            if not suffix.strip().startswith("."):
                raise ValueError(
                    f"allowed_suffixes 中的后缀必须以 '.' 开头喵! 当前后缀: {suffix}"
                )

        if self.resolver_name is not None:
            if not isinstance(self.resolver_name, str):
                raise TypeError(
                    f"resolver_name: {self.resolver_name} 必须为str或None类型的喵!"
                )
            if not self.resolver_name.strip():
                raise ValueError(
                    "resolver_name 不能为空字符串喵!"
                )

        if self.non_empty and self.kind != "directory":
            raise ValueError(
                "non_empty=True 时，kind 必须为 'directory' 喵!"
            )

    def normalize(self, value, path, context) -> Any:
        if value is MISSING:
            if self.default is not MISSING:
                return deepcopy(self.default)
            if self.required:
                raise MissingFieldError(
                    f"path: {path} 配置中缺少了该字段喵!"
                )
            return MISSING
        if value is None:
            if self.allow_none:
                return None
            raise NullNotAllowedError(
                f"path: {path}: {value} 该配置的值不能为空喵!"
            )
        if isinstance(value, pathlib.Path):
            return pathlib.Path(value)
        if isinstance(value, str):
            if not value.strip():
                raise SchemaError(
                    f"path: {path}: {value} 路径不能为空喵!"
                )
            try:
                return pathlib.Path(value)
            except (TypeError, ValueError, OSError) as e:
                raise TypeMismatchError(
                    f"path: {path}: {value} 该路径无法转换为pathlib.Path类型喵!"
                ) from e
        else:
            raise TypeMismatchError(
                f"path: {path}: {value} 必须为str或pathlib.Path类型的喵! "
                f"现在的type: {type(value)}"
            )