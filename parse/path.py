"""Parse 阶段的相对路径解析。

PARSE_REFACTOR_DEMO:
    本模块只把 Schema 已识别的 PathSchema 节点相对于 YAML 文件目录
    转换为绝对路径；不检查文件是否存在，也不创建任何运行时对象。
"""

from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path

from parse.context import MISSING
from parse.errors import PathResolutionError


def _resolve_path_values(value, base_path):
    """递归解析已被 Schema 标记为 pathlib.Path 的配置值。"""
    if isinstance(value, Path):
        if value.is_absolute():
            return value.resolve()
        return (base_path / value).resolve()
    if isinstance(value, dict):
        return {
            key: _resolve_path_values(child_value, base_path)
            for key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_path_values(child_value, base_path) for child_value in value]
    if isinstance(value, tuple):
        return tuple(_resolve_path_values(child_value, base_path) for child_value in value)
    return value


def resolve_paths(config, compiled_schema, context):
    """依据旧版 Parse 语义，将配置中的 Path 转换为绝对路径。

    PARSE_REFACTOR_DEMO:
        旧版先以 YAML 文件目录绝对化 Dataset root，再把 train/valid/test
        文件名拼接到 Dataset root。普通路径仍相对于 YAML 文件目录解析；
        Dataset split 采用专门规则，避免错误地落到 cfg 目录下。
    """
    if not isinstance(config, dict):
        raise PathResolutionError("待解析配置必须为dict类型的喵!")
    result = deepcopy(config)
    data_config = result.get("data")
    dataset_config = data_config.get("dataset") if isinstance(data_config, dict) else None
    original_splits = (
        deepcopy(dataset_config.get("splits"))
        if isinstance(dataset_config, dict)
        else None
    )
    result = _resolve_path_values(result, context.config_dir)
    if isinstance(original_splits, dict):
        resolved_dataset = result["data"]["dataset"]
        dataset_root = resolved_dataset.get("root")
        if isinstance(dataset_root, Path):
            for split_name, split_path in original_splits.items():
                split_path = Path(split_path)
                if split_path.is_absolute():
                    resolved_dataset["splits"][split_name] = split_path.resolve()
                else:
                    resolved_dataset["splits"][split_name] = (
                        dataset_root / split_path
                    ).resolve()
    return result
