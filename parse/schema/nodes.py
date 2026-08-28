from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
from copy import deepcopy
from collections.abc import Mapping

from parse.context import MISSING
from parse.errors import ConfigError, MissingFieldError, NullNotAllowedError, TypeMismatchError, \
    ParameterNormalizationError, UnknownFieldError, SchemaError


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

    def normalize(self, value, path, context) -> Any:
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
        if not self.choices:
            raise ValueError(f"choices 不能为空喵!")
        if isinstance(self.choices, Mapping):
            for choice_key, choice_value in self.choices.items():
                if not isinstance(choice_key, str):
                    raise ValueError(f"choices 中的键必须为str类型的喵! 当前key: {choice_key}, type: {type(choice_key)}")
                if not isinstance(choice_value, str):
                    raise ValueError(f"choices 中的值必须为str类型的喵! 当前value: {choice_value}, type: {type(choice_value)}")
                if not choice_key.strip():
                    raise ValueError(f"choices 中的键不能为空喵!")
                if not choice_value.strip():
                    raise ValueError(f"choices 中的值不能为空喵!")
        if isinstance(self.choices, set):
            for choice in self.choices:
                if not isinstance(choice, str):
                    raise ValueError(f"choices 中的元素必须为str类型的喵!")
                if not choice.strip():
                    raise ValueError(f"choices 中的元素不能为空喵!")
        if not type(self.case_sensitive) is bool:
            raise TypeError(f"case_sensitive: {self.case_sensitive} 必须为bool类型的喵! 当前的type: {type(self.case_sensitive)}")
        if not type(self.required) is bool:
            raise TypeError(f"required: {self.required} 必须为bool类型的喵! 当前的type: {type(self.required)}")
        if not type(self.allow_none) is bool:
            raise TypeError(f"allow_none: {self.allow_none} 必须为bool类型的喵! 当前的type: {type(self.allow_none)}")
        if self.required is True and self.default is not MISSING:
            raise ValueError(f"required 为True时，default必须为MISSING类型的喵! 现在default的type: {type(self.default)}")
