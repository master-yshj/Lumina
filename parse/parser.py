from pathlib import Path
from .errors import ConfigError
from .common import config_error
from .loader import (
    resolve_config_path, load_yaml, parse_cli_overrides, collect_origins,
    merge_config_sources, resolve_to_dict, read_run_header,
    load_training_projection, merge_inherited_config,
)
from .schema_registry import build_schema_catalog, select_root_schema
from .prepare import prepare_config
from .validate import validate_config, format_config_error


def parse_config(config_path: str | Path, cli_overrides: list[str] | None = None,
                 expected_mode: str | None = None) -> dict:
    """
    :param config_path: str | Path, 调用方提供的主 YAML 路径，相对路径以当前工作目录为起点
    :param cli_overrides: list[str] | None, 命令行覆盖列表，每项为“点分字段路径=值”，默认为 None，表示不覆盖
    :param expected_mode: str | None, 入口期望的 train、test、inference 或 visualize 模式，默认为 None，表示不额外限定入口模式

    :return: dict, 完成继承、默认值填充、规范化和校验后的普通配置字典，Path 和 tuple 等 Python 值保留

    该函数用于组织 YAML 与 CLI 合并、必要的训练配置继承和后续解析流程，返回供 Builder 使用的参数，不创建运行组件
    """
    try:
        yaml_path = resolve_config_path(config_path)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise config_error(f'主配置路径无效，请人工检查：{error}') from error
    file_config = load_yaml(yaml_path)
    file_origins = collect_origins(file_config, kind='yaml', source_path=yaml_path, base_dir=yaml_path.parent)
    cli_config = parse_cli_overrides([] if cli_overrides is None else cli_overrides)
    cli_origins = collect_origins(cli_config, kind='cli', source_path=None, base_dir=yaml_path.parent)
    current_config, current_origins = merge_config_sources([(file_config, file_origins), (cli_config, cli_origins)])
    catalog = build_schema_catalog()
    origins = current_origins
    root_schema = None
    try:
        mode, train_path = read_run_header(current_config, current_origins, yaml_path)
        if expected_mode is not None:
            if not isinstance(expected_mode, str) or expected_mode.strip().casefold() != mode:
                raise config_error(f'入口 expected_mode={expected_mode!r} 与 run.mode={mode!r} 不一致。', ('run', 'mode'))
        if mode == 'train':
            effective_config, origins = current_config, current_origins
        else:
            inherited_config, inherited_origins = load_training_projection(train_path, mode)
            effective_config, origins = merge_inherited_config(
                inherited_config, inherited_origins, current_config, current_origins, mode, catalog['task_specs'])
        raw_config, origins = resolve_to_dict(effective_config, origins)
        root_schema, task_key, mode = select_root_schema(raw_config, catalog, expected_mode)
        prepared_config, origins = prepare_config(raw_config, root_schema, origins, catalog, yaml_path.parent)
        config_model = validate_config(prepared_config, root_schema, origins, catalog)
    except ConfigError as error:
        if getattr(error, 'source_formatted', False):
            raise
        error_origins = getattr(error, 'config_origins', origins)
        raise format_config_error(error, error_origins, root_schema, catalog) from error
    return config_model.model_dump(mode='python')
