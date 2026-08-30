from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from parse.context import ConfigPath
from parse.schema.nodes import SchemaNode
from parse.schema.walk import SchemaEntry

@dataclass(frozen=True)
class CompiledSchema:
    root: SchemaNode                                        # 最终合并完成的根 Schema
    polymorphic_nodes: tuple[SchemaEntry,...]               # 最终 Schema 树中所有 PolymorphicSchema 对应的遍历条目
    path_nodes: tuple[SchemaEntry,...]                      # 最终 Schema 树中所有 PathSchema 对应的遍历条目
    validator_refs: tuple[Any, ...]                         # 编译过程中收集到的 Validator 引用
    schema_sources: Mapping[ConfigPath, tuple[str, ...]]    # 记录每个逻辑配置路径由哪些 Schema 来源共同贡献
