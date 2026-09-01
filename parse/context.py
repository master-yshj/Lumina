import pathlib
import typing

from dataclasses import dataclass
from collections.abc import Mapping

from .errors import BootstrapError

class _MissingType:
    def __repr__(self):
        """
        定义一个MissingType，用来校验那些本应存在但不存在的参数
        :return: "MISSING"
        """
        return "MISSING"

MISSING = _MissingType()

@dataclass(frozen=True)
class ConfigPath:
    """
    ConfigPath，用于构建配置路径的一个类，比如config.mode，有以下一些方法
    """
    parts: tuple[str | int, ...]

    def __post_init__(self):
        if not isinstance(self.parts, tuple):
            raise TypeError(f"parts: {self.parts} 必须为tuple类型的喵! 现在的type: {type(self.parts)}")
        if not self.parts:
            raise ValueError("parts 不能为空喵!")
        for part in self.parts:
            if isinstance(part, str):
                if not part.strip():
                    raise ValueError(f"parts: {self.parts} part: {part} 不能为空喵!")
            elif type(part) is int:
                if part < 0:
                    raise ValueError(f"parts: {self.parts} part: {part} 不能为负数喵!")
            else:
                raise TypeError(f"parts: {self.parts} 中的元素都必须为str或int类型的喵! part: {part} 现在的type: {type(part)}")

    @classmethod
    def root(cls, name: str) -> "ConfigPath":
        if not isinstance(name, str):
            raise TypeError(f"name(ConfigPath 根节点名称): {name} 必须为一个str类型的数据喵! 现在的type: {type(name)}")
        if not name.strip():
            raise ValueError(f"name(ConfigPath 根节点名称) 不能为空喵!")
        return cls(parts=(name,))

    def child(self, name: str) -> "ConfigPath":
        if not isinstance(name, str):
            raise TypeError(f"name(ConfigPath 子节点名称): {name} 必须为一个str类型的数据喵! 现在的type: {type(name)}")
        if not name.strip():
            raise ValueError(f"name(ConfigPath 子节点名称) 不能为空喵!")
        return ConfigPath(self.parts + (name.strip(),))

    def index(self, value: int) -> "ConfigPath":
        if not type(value) is int:
            raise TypeError(f"value(ConfigPath 中的列表索引): {value} 必须为一个int类型的数据喵! 现在的type: {type(value)}")
        if value < 0:
            raise ValueError(f"value(ConfigPath 中的列表索引) 不能小于0喵!")
        return ConfigPath(self.parts + (value,))

    def parent(self) -> "ConfigPath | None":
        if len(self.parts) <= 1:
            return None
        return ConfigPath(self.parts[:-1])

    def __str__(self) -> str:
        result = ""
        for part in self.parts:
            if isinstance(part, str):
                if result:
                    result += "."
                result += part
            else:
                result += f"[{part}]"
        return result

@dataclass(frozen=True)
class ComponentSelection:
    """
    ComponentSelection，用于收集元素的一个类，仅用于传参
    """
    category: str
    key: str
    path: ConfigPath
    spec: typing.Any

@dataclass(frozen=True)
class ParseContext:
    """
    ParseContext，用于获取某个键的元素的一个类
    """
    config_path: pathlib.Path
    config_dir: pathlib.Path
    expected_mode: str | None
    mode_key: str
    task_key: str
    selections: Mapping[ConfigPath, ComponentSelection]
    validate_runtime: bool

    def __post_init__(self):
        if not isinstance(self.config_path, pathlib.Path):
            raise TypeError(f"config_path: {self.config_path} 必须为pathlib.Path类型的喵! 现在的type: {type(self.config_path)}")
        if not isinstance(self.config_dir, pathlib.Path):
            raise TypeError(f"config_dir: {self.config_dir} 必须为pathlib.Path类型的喵! 现在的type: {type(self.config_dir)}")
        if isinstance(self.expected_mode, str):
            if not self.expected_mode.strip():
                raise ValueError(f"expected_mode 在为字符串类型时不能为空喵!")
        elif self.expected_mode is None:
            pass
        else:
            raise TypeError(f"expected_mode: {self.expected_mode} 必须为str类型或None喵! 现在的type: {type(self.expected_mode)}")
        if not isinstance(self.mode_key, str):
            raise TypeError(f"mode_key: {self.mode_key} 必须为str类型的喵! 现在的type: {type(self.mode_key)}")
        if not self.mode_key.strip():
            raise ValueError(f"mode_key 不能为空喵!")
        if not isinstance(self.task_key, str):
            raise TypeError(f"task_key: {self.task_key} 必须为str类型的喵! 现在的type: {type(self.task_key)}")
        if not self.task_key.strip():
            raise ValueError(f"task_key 不能为空喵!")
        if not isinstance(self.selections, Mapping):
            raise TypeError(f"selections: {self.selections} 必须为Mapping类型的喵! 现在的type: {type(self.selections)}")
        for selection_path, selection in self.selections.items():
            if not isinstance(selection_path, ConfigPath):
                raise TypeError(f"selections的键: {selection_path} 必须为ConfigPath类型的喵!")
            if not isinstance(selection, ComponentSelection):
                raise TypeError(f"selections[{selection_path}] 必须为ComponentSelection类型的喵!")
        if not type(self.validate_runtime) is bool:
            raise TypeError(f"validate_runtime: {self.validate_runtime} 必须为bool类型的喵! 现在的type: {type(self.validate_runtime)}")

    def get_selection(self, path: ConfigPath) -> ComponentSelection:
        if not path in self.selections:
            raise BootstrapError(f"path: {path} 该组件未被找到喵!")
        return self.selections[path]