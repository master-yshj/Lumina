from abc import ABC
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SchemaNode(ABC):
    """
    SchemaNode 抽象Schema节点基类
    """

    def normalize(self, value, path, context) -> Any:
        raise NotImplementedError