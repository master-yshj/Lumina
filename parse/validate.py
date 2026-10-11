from pathlib import Path
import stat
from typing import Annotated, Union, get_args, get_origin
from types import UnionType
from pydantic import BaseModel, ValidationError
from .errors import ConfigError
from .common import config_error, nearest_origin, format_field_path, source_description


def validate_model(prepared_config: dict, root_schema) -> BaseModel:
    """
    :param prepared_config: dict, 已经填好默认值并完成规范化的普通配置树
    :param root_schema: type[BaseModel], 当前任务和模式对应的根配置类

    :return: BaseModel, Pydantic 校验通过后创建的根配置实例，子配置块也成为对应的配置实例

    该函数用于从根配置类调用一次 model_validate，由 Pydantic 按嵌套字段定义检查类型、必填项、范围和多余字段
    """
    return root_schema.model_validate(prepared_config)


def validate_relations(config_model: BaseModel, catalog: dict) -> None:
    """
    :param config_model: BaseModel, 已通过 Pydantic 字段校验的根配置实例
    :param catalog: dict, build_schema_catalog 返回的规则集合，包含任务兼容范围、根配置类和 Union 分支规则

    :return: None, 所有跨字段关系检查通过后结束，不修改配置

    该函数用于检查任务与组件匹配、损失和指标约束、数据划分及长度、初始化与恢复互斥、设备精度及分布式组合、可视化索引范围
    """
    from torch.nn import ModuleDict
    cfg = config_model.model_dump(mode='python')
    task, mode = cfg['task']['type'], cfg['run']['mode']
    spec = catalog['task_specs'][task]
    if catalog['root_schemas'].get((task, mode)) is not type(config_model):
        raise config_error('所选根 Schema 与实际任务/模式不一致。', ('run', 'mode'))
    components = cfg['components']
    for category in ('tokenizer', 'model'):
        if components[category]['type'] not in spec[f'allowed_{category}s']:
            raise config_error(f'{task} 不支持该 {category}，允许 {sorted(spec[f"allowed_{category}s"])}。', ('components', category, 'type'))

    module_names = ModuleDict()

    def check_loss_list(losses, path):
        """
        :param losses: list[dict], 从配置实例导出的损失项列表，每项含 name、type、weight 和 params
        :param path: tuple[str | int, ...], 当前节点在配置树中的字段位置，不是磁盘路径

        :return: None, 当前损失列表检查通过后结束

        该函数用于读取外层 task 和空 ModuleDict 对象 module_names，检查名称重复及属性冲突、MLM 忽略标签是否为 -100，并要求至少一个损失权重大于 0
        """
        names = set()
        for index, loss in enumerate(losses):
            item_path = path + (index,)
            name = loss['name']
            if name in names:
                raise config_error(f'同一 Loss 集合中名称重复：{name!r}。', item_path + ('name',))
            if hasattr(module_names, name):
                raise config_error(f'Loss 名称 {name!r} 与 ModuleDict 属性冲突，请更换名称。', item_path + ('name',))
            names.add(name)
            if task == 'mlmpretrain' and loss['type'] == 'crossentropyloss' and loss['params']['ignore_index'] != -100:
                raise config_error('当前 MLM 标签忽略值固定为 -100，Loss 必须一致。', item_path + ('params', 'ignore_index'))
        if not any(loss['weight'] > 0 for loss in losses):
            raise config_error('每个 Loss 集合至少需要一个正权重。', path)

    if 'losses' in components:
        check_loss_list(components['losses'], ('components', 'losses'))
    output_fields = {'tokenaccuracy': {'token_accuracy'}, 'exactmatch': {'exact_match'}, 'characterf1': {'character_f1'}}
    for split, metrics in cfg.get('evaluation', {}).get('metrics', {}).items():
        names, outputs = set(), set()
        for index, metric in enumerate(metrics):
            path = ('evaluation', 'metrics', split, index)
            name = metric['type']
            if name not in spec['allowed_metrics']:
                raise config_error(f'指标 {name!r} 与任务 {task} 不匹配。', path + ('type',))
            if name in names or outputs & output_fields.get(name, set()):
                raise config_error('同一列表的指标重复或输出字段冲突。', path + ('type',))
            names.add(name)
            outputs.update(output_fields.get(name, set()))
            if task == 'mlmpretrain' and name == 'tokenaccuracy' and metric['params']['ignore_index'] != -100:
                raise config_error('MLM TokenAccuracy 的 ignore_index 必须为 -100。', path + ('params', 'ignore_index'))

    params = components['model']['params']
    if 'data' in cfg:
        data = cfg['data']
        if data['dataset']['type'] not in spec['dataset_types']:
            raise config_error('Dataset 类型与任务不匹配。', ('data', 'dataset', 'type'))
        if data['collator']['type'] != spec['collator_key']:
            raise config_error('Collator 类型与任务不匹配。', ('data', 'collator', 'type'))
        if task == 'mlmpretrain' and data['dataset']['params']['max_seq_length'] > params['max_position_embeddings']:
            raise config_error('数据 max_seq_length 超过模型 max_position_embeddings。', ('data', 'dataset', 'params', 'max_seq_length'))
        needed = [mode]
        if mode == 'train':
            validation = cfg['optimization']['validation']
            if validation['enabled']:
                needed.append('valid')
            elif validation['early_stopping']['enabled']:
                raise config_error('关闭 validation 时须显式关闭 early_stopping。', ('optimization', 'validation', 'early_stopping', 'enabled'))
            if cfg['evaluation']['test_after_train']:
                needed.append('test')
        for split in needed:
            if data['dataset']['splits'].get(split) is None:
                raise config_error(f'当前动作要求 {split} 数据文件。', ('data', 'dataset', 'splits', split))
            if data['loaders'].get(split) is None:
                raise config_error(f'当前动作要求 {split} loader 配置块，允许填写空字典使用默认值。', ('data', 'loaders', split))
    if mode == 'inference':
        inference = cfg['inference']
        if inference['decoding']['type'] not in spec['allowed_decodings']:
            raise config_error('解码器与当前任务不匹配。', ('inference', 'decoding', 'type'))
        if task == 'mlmpretrain' and inference['max_seq_length'] > params['max_position_embeddings']:
            raise config_error('推理 max_seq_length 超过模型的位置上限。', ('inference', 'max_seq_length'))
    if mode == 'train':
        if cfg['run']['resume_from'] is not None and components['model']['init_from'] is not None:
            raise config_error('resume_from 与 model.init_from 不能同时设置。', ('run', 'resume_from'))
        scheduler = cfg['optimization']['scheduler']
        if scheduler and scheduler['type'] == 'warmup':
            if scheduler['params']['base_scheduler_type'] not in catalog['component_specs']['base_scheduler_specs']:
                raise config_error('基础 Scheduler 上下文未提供此参数规则。', ('optimization', 'scheduler', 'params', 'base_scheduler_type'))
    elif components['model']['init_from'] is not None:
        raise config_error('非训练应从当前 checkpoint 加载，init_from 必须由继承政策禁用。', ('components', 'model', 'init_from'))

    runtime = cfg['runtime']
    device, distributed = runtime['device'], runtime['distributed']
    if device['type'] == 'cpu' and device['index'] is not None:
        raise config_error('CPU 不使用设备编号，index 必须为 null。', ('runtime', 'device', 'index'))
    if device['type'] == 'cpu' and runtime['precision'] == 'fp16':
        raise config_error('首期不支持 CPU FP16；请选择 fp32 或经运行检查的 bf16。', ('runtime', 'precision'))
    if distributed['strategy'] == 'none':
        if distributed['backend'] is not None:
            raise config_error('单进程 strategy=none 时 backend 必须为 null。', ('runtime', 'distributed', 'backend'))
    else:
        for valid, path, message in [
            (mode == 'train', ('run', 'mode'), '首期 DDP 仅支持训练模式。'),
            (device['type'] == 'cuda', ('runtime', 'device', 'type'), '首期 DDP 仅支持单机 CUDA 多卡。'),
            (device['index'] is None, ('runtime', 'device', 'index'), 'DDP 的设备编号来自 LOCAL_RANK，index 必须为 null。'),
            (distributed['backend'] == 'nccl', ('runtime', 'distributed', 'backend'), '首期 CUDA DDP 要求 NCCL。'),
        ]:
            if not valid:
                raise config_error(message, path)
    for index, method in enumerate(cfg.get('visualize', {}).get('methods', [])):
        path = ('visualize', 'methods', index, 'params')
        if method['type'] == 'losslandscape':
            check_loss_list(method['params']['losses'], path + ('losses',))
        elif method['type'] == 'attentionheatmap':
            for field, limit in [('layer', params['num_layers']), ('head', params['num_heads'])]:
                selection = method['params'][field]
                if selection == 'all':
                    continue
                indices = selection if isinstance(selection, list) else [selection]
                for offset, value in enumerate(indices):
                    if not -limit <= value < limit:
                        item_path = path + (field,) + ((offset,) if isinstance(selection, list) else ())
                        raise config_error(f'索引 {value} 越界，要求 {-limit} <= index < {limit}。', item_path)


def validate_resources(config_model: BaseModel) -> None:
    """
    :param config_model: BaseModel, 已通过字段和关系校验的根配置实例

    :return: None, 本次运行需要的资源检查通过后结束

    该函数用于依据 resource 标记检查路径存在性、文件或目录类型及规定后缀；忽略 metadata、None 和未启用的数据划分，不读取文件内容或加载权重
    """
    cfg = config_model.model_dump(mode='python')
    mode = cfg['run']['mode']
    active_splits = {mode}
    if mode == 'train':
        if cfg['optimization']['validation']['enabled']:
            active_splits.add('valid')
        if cfg['evaluation']['test_after_train']:
            active_splits.add('test')

    def check_path(value, path, kind):
        """
        :param value: Path, 准备阶段产生的资源绝对路径
        :param path: tuple[str | int, ...], 当前节点在配置树中的字段位置，不是磁盘路径
        :param kind: str, 资源种类，当前包括 directory、file、checkpoint 和 config

        :return: None, 当前路径符合资源要求后结束

        该函数为validate_resources的内部函数用于通过 Path.stat 检查资源是否存在且可访问，再检查目录或普通文件类型；checkpoint 要求 .pt，config 要求 .yaml 或 .yml
        """
        if not isinstance(value, Path) or not value.is_absolute():
            raise config_error('资源必须已经规范为绝对 Path。', path)
        try:
            status = value.stat()
        except FileNotFoundError as error:
            raise config_error(f'资源不存在：{value}', path) from error
        except OSError as error:
            raise config_error(f'资源无法访问：{value}；系统原因：{error}', path) from error
        is_valid = stat.S_ISDIR(status.st_mode) if kind == 'directory' else stat.S_ISREG(status.st_mode)
        if not is_valid:
            raise config_error(f'{value} 必须指向' + ('目录。' if kind == 'directory' else '普通文件。'), path)
        suffixes = {'checkpoint': {'.pt'}, 'config': {'.yaml', '.yml'}}
        if kind in suffixes and value.suffix.casefold() not in suffixes[kind]:
            raise config_error(f'{value} 的后缀应为 {sorted(suffixes[kind])}。', path)

    def walk(model, path):
        """
        :param model: object, 当前配置实例、list、tuple 或字段值
        :param path: tuple[str | int, ...], 当前节点在配置树中的字段位置，不是磁盘路径

        :return: None, 当前节点及其需要检查的子节点遍历完成后结束

        该函数为validate_resources的内部函数用于读取字段的 resource 标记并调用 check_path，按外层 active_splits 跳过未启用划分，同时把 run.resume_from 作为 checkpoint 检查
        """
        if isinstance(model, BaseModel):
            for name, field in type(model).model_fields.items():
                child_path = path + (name,)
                value = getattr(model, name)
                if value is None:
                    continue
                if path == ('data', 'dataset', 'splits') and name not in active_splits:
                    continue
                kind = (field.json_schema_extra or {}).get('resource')
                if child_path == ('run', 'resume_from'):
                    kind = 'checkpoint'
                if kind and kind != 'metadata':
                    check_path(value, child_path, kind)
                walk(value, child_path)
        elif isinstance(model, (list, tuple)):
            for index, child in enumerate(model):
                walk(child, path + (index,))
    walk(config_model, ())


def _business_location(location: tuple, root_schema, catalog: dict) -> tuple:
    """
    :param location: tuple, Pydantic 错误记录中的 loc，可能混有 Union 分支名称
    :param root_schema: type[BaseModel], 当前任务和模式对应的根配置类
    :param catalog: dict, build_schema_catalog 返回的规则集合，包含任务兼容范围、根配置类和 Union 分支规则

    :return: tuple, 去除类型分支标签后的实际配置字段路径，保留业务字段名和列表下标

    该函数用于沿配置类的真实字段及分支规则解释错误位置，使后续错误提示能对应 YAML 中的字段和来源记录
    """
    annotation = root_schema
    result = []
    index = 0
    while index < len(location):
        while get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        if get_origin(annotation) in (Union, UnionType):
            args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
            reduced = args[0] if len(args) == 1 else Union[args]
            route = catalog['union_routes'].get(reduced)
            if route and location[index] in route['variants']:
                annotation = route['variants'][location[index]]
                index += 1
                continue
            if len(args) == 1:
                annotation = args[0]
                continue
        part = location[index]
        result.append(part)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel) and part in annotation.model_fields:
            annotation = annotation.model_fields[part].annotation
        elif get_origin(annotation) is list:
            annotation = get_args(annotation)[0]
        elif get_origin(annotation) is tuple and isinstance(part, int):
            args = get_args(annotation)
            annotation = args[part] if part < len(args) else None
        else:
            result.extend(location[index + 1:])
            break
        index += 1
    return tuple(result)


def format_config_error(error, origins, root_schema, catalog) -> ConfigError:
    """
    :param error: Exception, Pydantic ValidationError 或带 field_path 的配置错误，已格式化错误会原样返回
    :param origins: dict, 键为字段位置元组，值为 YAML、CLI 或默认值的来源信息
    :param root_schema: type[BaseModel], 当前任务和模式对应的根配置类
    :param catalog: dict, build_schema_catalog 返回的规则集合，包含任务兼容范围、根配置类和 Union 分支规则

    :return: ConfigError, 包含字段位置、错误原因及来源说明的异常对象，由调用方抛出

    该函数用于将校验错误的位置转换为实际配置路径并附上最近来源，标记 source_formatted 以避免重复包装，此函数只构造或返回异常
    """
    if getattr(error, 'source_formatted', False):
        return error
    if isinstance(error, ValidationError):
        records = [(_business_location(tuple(item['loc']), root_schema, catalog), item['msg']) for item in error.errors()]
    else:
        records = [(getattr(error, 'field_path', ()), str(error))]
    messages = [f'{format_field_path(path)}：{message}\n  来源：{source_description(nearest_origin(origins, path))}' for path, message in records]
    result = config_error('配置校验失败，请人工检查以下字段：\n' + '\n'.join(messages), records[0][0])
    result.source_formatted = True
    return result


def validate_config(prepared_config, root_schema, origins, catalog) -> BaseModel:
    """
    :param prepared_config: dict, 已填默认值且规范化完成的普通配置树
    :param root_schema: type[BaseModel], 当前任务和模式对应的根配置类
    :param origins: dict, 包含原始字段和新增默认字段的来源表
    :param catalog: dict, build_schema_catalog 返回的规则集合，包含任务兼容范围、根配置类和 Union 分支规则

    :return: BaseModel, 字段、关系和资源检查全部通过后的根配置实例

    该函数用于依次执行 Pydantic 字段校验、跨字段关系校验及资源校验，将失败信息包装为包含字段来源的 ConfigError
    """
    try:
        model = validate_model(prepared_config, root_schema)
        validate_relations(model, catalog)
        validate_resources(model)
    except (ValidationError, ConfigError) as error:
        raise format_config_error(error, origins, root_schema, catalog) from error
    return model
