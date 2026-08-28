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
        return ConfigPath(self.parts + (name,))

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

    def get_selection(self, path: ConfigPath) -> ComponentSelection:
        if not path in self.selections:
            raise BootstrapError(f"path: {path} 该组件未被找到喵!")
        return self.selections[path]