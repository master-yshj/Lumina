from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from parse.context import ConfigPath
from parse.schema.nodes import PathSchema, PolymorphicSchema, SchemaNode
from parse.specs.validator import ValidatorSpec
from parse.schema.walk import SchemaEntry, walk_schema

@dataclass(frozen=True)
class CompiledSchema:
    root: SchemaNode                                        # 最终合并完成的根 Schema
    polymorphic_nodes: tuple[SchemaEntry,...]               # 最终 Schema 树中所有 PolymorphicSchema 对应的遍历条目
    path_nodes: tuple[SchemaEntry,...]                      # 最终 Schema 树中所有 PathSchema 对应的遍历条目
    validator_refs: tuple[Any, ...]                         # 编译过程中收集到的 Validator 引用
    schema_sources: Mapping[ConfigPath, tuple[str, ...]]    # 记录每个逻辑配置路径由哪些 Schema 来源共同贡献


def compile_schema(root, validator_refs=(), schema_sources=None):
    """遍历最终 Schema 并生成 Parse 执行阶段使用的索引。

    PARSE_REFACTOR_DEMO:
        当前编译器已经使用正式职责边界，但 validator_refs 和来源信息仍由
        调用方提供；后续 Task Schema contribution 稳定后再自动收集。
    """
    if not isinstance(root, SchemaNode):
        raise TypeError(f"root: {root} 必须为SchemaNode类型的喵! 现在的type: {type(root)}")
    if not isinstance(validator_refs, tuple):
        raise TypeError(f"validator_refs: {validator_refs} 必须为tuple类型的喵!")
    if not all(isinstance(validator, ValidatorSpec) for validator in validator_refs):
        raise TypeError("validator_refs 中的元素必须为ValidatorSpec类型的喵!")
    if schema_sources is None:
        schema_sources = {}
    if not isinstance(schema_sources, Mapping):
        raise TypeError(f"schema_sources: {schema_sources} 必须为Mapping类型的喵!")

    entries = tuple(walk_schema(root))
    polymorphic_nodes = tuple(
        entry for entry in entries if isinstance(entry.node, PolymorphicSchema)
    )
    path_nodes = tuple(
        entry for entry in entries if isinstance(entry.node, PathSchema)
    )
    return CompiledSchema(
        root=root,
        polymorphic_nodes=polymorphic_nodes,
        path_nodes=path_nodes,
        validator_refs=validator_refs,
        schema_sources=dict(schema_sources),
    )
