from dataclasses import dataclass
from collections.abc import Iterable

from parse.schema.nodes import SchemaNode, MappingSchema, PolymorphicSchema, SequenceSchema
from parse.context import ConfigPath

@dataclass(frozen=True)
class SchemaEntry:
    """
    SchemaEntry
    """
    path: ConfigPath
    node: SchemaNode

def walk_schema(root_node, root_path=ConfigPath.root("config")):
    """
    :param root_node: SchemaNode 当前准备遍历的SchemaNode
    :param root_path: ConfigPath 当前root所在配置树中的位置，默认值为ConfigPath.root("config")
    """
    curr_node = SchemaEntry(root_path, root_node)
    if not isinstance(curr_node.node, SchemaNode):
        raise TypeError(f"curr_node: {curr_node} 的node应该为SchemaNode类型的喵! 现在的type: {type(curr_node.node)}")
    if not isinstance(curr_node.path, ConfigPath):
        raise TypeError(f"curr_node: {curr_node} 的path应该为ConfigPath类型的喵! 现在的type: {type(curr_node.path)}")
    yield curr_node
    if isinstance(root_node, MappingSchema):
        for field_name, field_schema in root_node.fields.items():
            child_path = root_path.child(field_name)
            yield from walk_schema(field_schema, child_path)
    if isinstance(root_node, SequenceSchema):
        item_schema = root_node.item_schema
        item_path = curr_node.path.child("<item>")
        yield from walk_schema(item_schema, item_path)
    if isinstance(root_node, PolymorphicSchema):
        common_schema = root_node.common_schema
        yield from walk_schema(common_schema, root_path)
