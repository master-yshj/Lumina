import pathlib

from collections.abc import Mapping, Sequence

from parse.context import ComponentSelection, ConfigPath, ParseContext
from parse.errors import BootstrapError, MissingFieldError, TypeMismatchError

def _read_string(mapping, field_name, path):
    if field_name not in mapping:
        raise MissingFieldError(f"path: {path.child(field_name)} 缺少必填字段喵!")
    value = mapping[field_name]
    if not isinstance(value, str):
        raise TypeMismatchError(
            f"path: {path.child(field_name)}: {value} 必须为str类型的喵! "
            f"现在的type: {type(value)}"
        )
    if not value.strip():
        raise BootstrapError(f"path: {path.child(field_name)} 的值不能为空喵!")
    return value


def _scan_path(raw_config, descriptor, root_path):
    matches = []

    def visit(current_value, pattern_index, current_path):
        if pattern_index == len(descriptor.path_pattern):
            matches.append((current_path, current_value))
            return

        part = descriptor.path_pattern[pattern_index]
        if part == "*":
            if isinstance(current_value, Mapping):
                for field_name, child_value in current_value.items():
                    if not isinstance(field_name, str):
                        raise TypeMismatchError(
                            f"path: {current_path}: Mapping字段名必须为str类型的喵! "
                            f"当前字段名: {field_name}"
                        )
                    visit(
                        child_value,
                        pattern_index + 1,
                        current_path.child(field_name),
                    )
            elif isinstance(current_value, Sequence) and not isinstance(
                current_value, (str, bytes, bytearray)
            ):
                for index, child_value in enumerate(current_value):
                    visit(
                        child_value,
                        pattern_index + 1,
                        current_path.index(index),
                    )
            else:
                raise TypeMismatchError(
                    f"path: {current_path}: 通配符只能用于Mapping或list，"
                    f"当前type: {type(current_value)}"
                )
            return

        if not isinstance(current_value, Mapping):
            raise TypeMismatchError(
                f"path: {current_path}: 继续读取字段前当前值必须为Mapping类型的喵! "
                f"现在的type: {type(current_value)}"
            )
        if part not in current_value:
            return
        visit(
            current_value[part],
            pattern_index + 1,
            current_path.child(part),
        )

    visit(raw_config, 0, root_path)
    if not matches and descriptor.required:
        missing_path = root_path
        for part in descriptor.path_pattern:
            if part == "*":
                break
            missing_path = missing_path.child(part)
        raise MissingFieldError(
            f"path: {missing_path}: 缺少Bootstrap要求的配置节点喵!"
        )
    return matches


def _create_selection(node_path, node_value, descriptor, runtime_registry, spec_registry):
    if not isinstance(node_value, Mapping):
        raise TypeMismatchError(
            f"path: {node_path}: 组件节点必须为Mapping类型的喵! "
            f"现在的type: {type(node_value)}"
        )
    raw_type = _read_string(node_value, "type", node_path)
    canonical_key = runtime_registry.resolve_key(raw_type)
    spec = spec_registry.get(canonical_key)
    if spec is None:
        raise BootstrapError(f"path: {node_path} 的ComponentSpec不能为空喵!")
    if getattr(spec, "category", None) != descriptor.category:
        raise BootstrapError(
            f"path: {node_path} 的ComponentSpec类别不一致喵! "
            f"spec.category: {getattr(spec, 'category', None)} "
            f"category: {descriptor.category}"
        )
    if getattr(spec, "key", None) != canonical_key:
        raise BootstrapError(
            f"path: {node_path} 的ComponentSpec键不一致喵! "
            f"spec.key: {getattr(spec, 'key', None)} canonical_key: {canonical_key}"
        )
    return ComponentSelection(
        category=descriptor.category,
        key=canonical_key,
        path=node_path,
        spec=spec,
    )


def _derive_mode(task_key):
    if "_" not in task_key:
        raise BootstrapError(
            f"task_key: {task_key} 必须包含用于派生mode的下划线喵!"
        )
    _, mode_key = task_key.rsplit("_", 1)
    if not mode_key:
        raise BootstrapError(
            f"task_key: {task_key} 的mode不能为空喵!"
        )
    return mode_key


def bootstrap_config(
    raw_config,
    config_path,
    task_registry,
    task_spec_registry,
    component_registries,
    component_spec_registries,
    expected_mode=None,
    validate_runtime=True,
):
    if not isinstance(raw_config, Mapping):
        raise BootstrapError(
            f"raw_config: {raw_config} 必须为Mapping类型的喵! "
            f"现在的type: {type(raw_config)}"
        )
    if not isinstance(config_path, pathlib.Path):
        config_path = pathlib.Path(config_path)
    config_path = config_path.absolute()
    config_dir = config_path.parent

    root_path = ConfigPath.root("config")
    raw_task = _read_string(raw_config, "task", root_path)
    task_key = task_registry.resolve_key(raw_task)
    task_builder = task_registry.get(task_key)
    task_spec = task_spec_registry.get(task_key)
    if task_spec is None:
        raise BootstrapError(f"task_key: {task_key} 对应的Task Spec不能为空喵!")
    if getattr(task_spec, "category", None) != "task":
        raise BootstrapError(
            f"task_key: {task_key} 对应的Spec类别不正确喵! "
            f"spec.category: {getattr(task_spec, 'category', None)}"
        )
    if getattr(task_spec, "key", None) != task_key:
        raise BootstrapError(
            f"task_key: {task_key} 与Task Spec key: {getattr(task_spec, 'key', None)} 不一致喵!"
        )
    if getattr(task_spec, "factory", None) is not task_builder:
        raise BootstrapError(
            f"task_key: {task_key} 的Task Spec factory与Task Registry不一致喵!"
        )

    mode_key = _derive_mode(task_key)
    if expected_mode is not None:
        if not isinstance(expected_mode, str):
            raise TypeError(
                f"expected_mode: {expected_mode} 必须为str类型或None喵! "
                f"现在的type: {type(expected_mode)}"
            )
        normalized_expected_mode = expected_mode.strip().casefold()
        if not normalized_expected_mode:
            raise ValueError("expected_mode 不能为空喵!")
        if normalized_expected_mode != mode_key:
            raise BootstrapError(
                f"expected_mode: {normalized_expected_mode} 与mode_key: {mode_key} 不一致喵!"
            )

    descriptors = getattr(task_spec, "bootstrap_descriptors", None)
    if not isinstance(descriptors, tuple):
        raise BootstrapError(
            f"task_key: {task_key} 的Task Spec必须提供tuple类型的bootstrap_descriptors喵!"
        )

    selections = {}
    for descriptor in descriptors:
        runtime_registry = component_registries.get(descriptor.category)
        spec_registry = component_spec_registries.get(descriptor.category)
        if runtime_registry is None or spec_registry is None:
            raise BootstrapError(
                f"category: {descriptor.category} 缺少运行Registry或Spec Registry喵!"
            )
        matches = _scan_path(raw_config, descriptor, root_path)
        for node_path, node_value in matches:
            if node_path in selections:
                raise BootstrapError(f"path: {node_path} 被重复选择喵!")
            selections[node_path] = _create_selection(
                node_path,
                node_value,
                descriptor,
                runtime_registry,
                spec_registry,
            )

    return ParseContext(
        config_path=config_path,
        config_dir=config_dir,
        expected_mode=(
            None
            if expected_mode is None
            else expected_mode.strip().casefold()
        ),
        mode_key=mode_key,
        task_key=task_key,
        selections=selections,
        validate_runtime=validate_runtime,
    )
