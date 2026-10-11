from pathlib import Path
from typing import Annotated, Literal, Union, get_args, get_origin
from pydantic import Field, create_model
from . import schemas as s
from .common import config_error

MODES = ('train', 'test', 'inference', 'visualize')


def load_runtime_registries() -> dict:
    """
    :return: dict, 键为 tokenizer、model、loss、optimizer、scheduler、metric、decoding、visualizer，值为各类别的 Registry 对象

    该函数用于调用时导入正式组件注册表，供解析器检查组件名称并解析别名，不实例化组件
    """
    from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
    from models.Transformer.modeling.registry import MODEL_REGISTRY
    from models.Transformer.losses.registry import LOSS_REGISTRY
    from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
    from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
    from metrics.registry import METRIC_REGISTRY
    from Inference.decoding.registry import DECODING_REGISTRY
    from visualize.registry import VISUALIZER_REGISTRY
    return dict(tokenizer=TOKENIZER_REGISTRY, model=MODEL_REGISTRY, loss=LOSS_REGISTRY,
                optimizer=OPTIMIZER_REGISTRY, scheduler=SCHEDULER_REGISTRY,
                metric=METRIC_REGISTRY, decoding=DECODING_REGISTRY, visualizer=VISUALIZER_REGISTRY)


def make_component_variant(class_name, canonical_key, params_schema, shell_schema, category):
    """
    :param class_name: str, 动态创建的配置类名称
    :param canonical_key: str, 组件的规范名，也是 type 字段唯一允许的值
    :param params_schema: object, params 使用的配置类或带分支选择规则的类型注解
    :param shell_schema: type[ConfigBase], 新配置类继承的基类，提供公共字段和校验规则
    :param category: str, 组件所属注册类别，例如 tokenizer，用于名称规范化时查找对应 Registry

    :return: type[ConfigBase], 继承 shell_schema 并声明 type、params 字段的新配置类

    该函数用于使用 Pydantic.create_model 声明组件结构；params 默认从空字典开始，后续准备阶段再填参数默认值，此处不创建组件或配置实例
    """
    fields = {
        'type': (Literal[canonical_key], Field(json_schema_extra={'normalize': 'name', 'registry': category})),
        'params': (params_schema, Field(default_factory=dict)),
    }
    return create_model(class_name, __base__=shell_schema, **fields)


def make_choice_type(variants: dict, discriminator: str, union_routes: dict):
    """
    :param variants: dict[str, type[ConfigBase]], 键为规范名，值为该名称对应的配置类
    :param discriminator: str, 用于选择分支的字段名，例如 type 或 base_scheduler_type
    :param union_routes: dict, 键为 Union 类型对象，值含 discriminator 字段名和 variants 名称到配置类的映射

    :return: object, 单分支返回配置类，多分支返回带 Field(discriminator=...) 的 Annotated Union 类型注解

    该函数用于核对每个分支的 Literal 值；多分支时原地更新 union_routes，供准备器选类，同时把联合类型交给 Pydantic 校验
    """
    if not variants:
        raise config_error('Schema 分支不能为空喵! 请人工检查组件登记喵~')
    for key, schema in variants.items():
        if get_args(schema.model_fields[discriminator].annotation) != (key,):
            raise config_error(f'{schema.__name__}.{discriminator} 与规范名 {key} 不一致喵!')
    if len(variants) == 1:
        return next(iter(variants.values()))
    union_type = Union[tuple(variants.values())]
    union_routes[union_type] = {'discriminator': discriminator, 'variants': dict(variants)}
    return Annotated[union_type, Field(discriminator=discriminator)]


def build_component_types(union_routes: dict) -> tuple[dict, dict]:
    """
    :param union_routes: dict, 保存 Union 类型到分支选择规则的映射，本函数通过 make_choice_type 向其中添加记录

    :return: tuple[dict, dict], 按顺序返回 component_specs 参数规则表和 category_types 类别到组件配置类型的映射

    该函数用于登记当前支持的组件参数类，按依赖顺序生成组件类型、Warmup 参数和 Loss Landscape 参数；component_specs 还保存 base_scheduler_specs 基础调度器参数表
    """
    component_specs = {}
    category_types = {}

    def register_category(category, entries, shell):
        """
        :param category: str, 要登记的组件类别，例如 optimizer
        :param entries: dict, 键为组件规范名，值为 params 配置类或可选择多个配置类的类型注解
        :param shell: type[ConfigBase], 当前类别的组件配置基类

        :return: None, 结果写入外层 component_specs、category_types，多个分支时还更新 union_routes

        该函数为build_component_types的内部函数, 用于为当前类别逐项生成组件配置类并组合成可选择的类型，记录名称、参数规则和公共基类的对应关系
        """
        records = {key: {'params_schema': params, 'shell_schema': shell} for key, params in entries.items()}
        component_specs[category] = records
        variants = {
            key: make_component_variant(f'{category}_{key}Config', key, record['params_schema'], shell, category)
            for key, record in records.items()
        }
        category_types[category] = make_choice_type(variants, 'type', union_routes)

    register_category('tokenizer', dict(exampletokenizer=s.TokenizerParamsConfig, qatokenizer=s.TokenizerParamsConfig), s.ComponentCommonConfig)
    register_category('model', dict(tinybertformaskedlm=s.TinyBertParamsConfig, tinybertforquestionanswering=s.TinyBertParamsConfig), s.ModelCommonConfig)
    register_category('loss', dict(crossentropyloss=s.CrossEntropyParamsConfig), s.LossCommonConfig)
    register_category('optimizer', dict(adamw=s.AdamWParamsConfig), s.ComponentCommonConfig)
    register_category('metric', dict(tokenaccuracy=s.TokenAccuracyParamsConfig, exactmatch=s.EmptyParamsConfig, characterf1=s.EmptyParamsConfig), s.ComponentCommonConfig)
    register_category('decoding', dict(topksampling=s.TopKParamsConfig, qaspantopkdecoding=s.QASpanParamsConfig), s.ComponentCommonConfig)

    base_specs = {'steplr': s.BaseStepLRParamsConfig}
    component_specs['base_scheduler_specs'] = base_specs
    variants = {}
    for key, params in base_specs.items():
        variants[key] = create_model(
            f'Warmup_{key}ParamsConfig', __base__=s.WarmupCommonConfig,
            base_scheduler_type=(Literal[key], Field(json_schema_extra={'normalize': 'name', 'registry': 'scheduler'})),
            base_scheduler_params=(params, Field(default_factory=dict)),
        )
    warmup_params = make_choice_type(variants, 'base_scheduler_type', union_routes)
    register_category('scheduler', dict(steplr=s.StepLRParamsConfig, warmup=warmup_params), s.SchedulerCommonConfig)
    landscape_params = create_model(
        'LossLandscapeParamsConfig', __base__=s.LossLandscapeCommonConfig,
        losses=(list[category_types['loss']], Field(min_length=1)),
    )
    register_category('visualizer', dict(attentionheatmap=s.AttentionHeatmapParamsConfig, losslandscape=landscape_params), s.ComponentCommonConfig)
    return component_specs, category_types


def check_component_bindings(runtime_registries: dict, component_specs: dict) -> None:
    """
    :param runtime_registries: dict, 键为组件类别，值为对应的运行 Registry
    :param component_specs: dict, build_component_types 返回的参数规则表，含各类别规则和 base_scheduler_specs

    :return: None, 登记对应关系检查通过后结束

    该函数用于对解析器支持的组件规范名能否在运行注册表中找到对象，以及 params 是否使用有效配置类；运行注册表额外存在的名称不会因此自动获得解析规则
    """
    def valid_schema(annotation):
        """
        :param annotation: object, 待检查的 params 类型注解

        :return: bool, 配置类继承 ConfigBase，或 Annotated、Union 包装内的分支均满足要求时为 True

        该函数为check_component_bindings的内部函数, 用于递归识别有效的配置类型，避免组件参数规则退化成任意 dict
        """
        if get_origin(annotation) is Annotated:
            return valid_schema(get_args(annotation)[0])
        if get_origin(annotation) is Union:
            return all(valid_schema(arg) for arg in get_args(annotation))
        return isinstance(annotation, type) and issubclass(annotation, s.ConfigBase)

    for category, registry in runtime_registries.items():
        for key, record in component_specs[category].items():
            if registry.resolve_key(key) != key or registry.get(key) is None:
                raise config_error(f'{category}.{key} 的运行登记不正确，请人工检查 Registry。')
            if not valid_schema(record['params_schema']):
                raise config_error(f'{category}.{key} 未提供有效参数 Schema，不能使用任意 dict。')
    for key in component_specs['base_scheduler_specs']:
        if runtime_registries['scheduler'].resolve_key(key) != key:
            raise config_error(f'基础 Scheduler {key} 必须使用规范名。')
        runtime_registries['scheduler'].get(key)


def resolve_task_key(value, task_specs: dict, field_path=('task', 'type')) -> str:
    if not isinstance(value, str) or not value.strip():
        raise config_error('task.type 必须为非空字符串。', field_path)

    from tasks.registry import TASK_BUILDER_REGISTRY
    from .errors import UnknownRegistryKeyError

    try:
        canonical_key = TASK_BUILDER_REGISTRY.resolve_key(value)
    except UnknownRegistryKeyError as error:
        raise config_error(f'任务名称 {value!r} 未注册，请人工检查任务注册表。', field_path) from error

    if canonical_key not in task_specs:
        raise config_error(f'任务 {canonical_key!r} 已注册，但尚未提供配置解析规则，请人工检查任务规则表。', field_path)
    return canonical_key


def build_schema_catalog() -> dict:
    """
    :return: dict, 包含 runtime_registries、component_specs、category_types、task_specs、root_schemas、union_routes 六张规则表

    该函数用于组织运行注册表和参数规则，为各任务的 train、test、inference、visualize 生成根配置类；每次调用建立独立规则集合，不读取 YAML
    """
    union_routes = {}
    registries = load_runtime_registries()
    specs, types = build_component_types(union_routes)
    check_component_bindings(registries, specs)
    from tasks.registry import TASK_BUILDER_REGISTRY

    tasks = {}
    for task_key in TASK_BUILDER_REGISTRY.keys():
        tasks[task_key] = TASK_BUILDER_REGISTRY.get_spec(task_key)
    roots = {}

    def make_fields_model(class_name, base_schema, fields):
        """
        :param class_name: str, 动态配置类的名称
        :param base_schema: type[ConfigBase], 新类继承的配置基类
        :param fields: dict, 键为字段名，值为 (类型注解, Field 定义或默认值) 二元组

        :return: type[ConfigBase], Pydantic.create_model 创建的配置类对象

        该函数用于把当前配置块的字段定义组合成配置类，供外层构建根配置结构，此处不创建配置实例
        """
        return create_model(class_name, __base__=base_schema, **fields)

    for task_key, spec in tasks.items():
        dataset = make_fields_model(f'{task_key}Dataset', s.ConfigBase, {
            'type': (Literal[spec['dataset_types']], Field(json_schema_extra={'normalize': 'name'})),
            'root': (Path, Field(json_schema_extra={'normalize': 'path', 'resource': 'directory'})),
            'splits': (s.DatasetSplitsConfig, Field()),
            'params': (spec['dataset_params_schema'], Field(default_factory=dict)),
        })
        collator = make_fields_model(f'{task_key}Collator', s.ConfigBase, {
            'type': (Literal[spec['collator_key']], Field(json_schema_extra={'normalize': 'name'})),
            'params': (spec['collator_params_schema'], Field(default_factory=dict)),
        })
        for mode in spec['modes']:
            prefix = f'{task_key}_{mode}'
            component_fields = {key: (types[key], Field()) for key in ('tokenizer', 'model')}
            if mode in ('train', 'test'):
                component_fields['losses'] = (list[types['loss']], Field(min_length=1))
            components = make_fields_model(prefix+'Components', s.ConfigBase, component_fields)
            fields = {'run': (s.TrainRunConfig if mode == 'train' else s.CheckpointRunConfig, Field()),
                      'components': (components, Field())}
            if mode != 'inference':
                if mode == 'train':
                    loader_fields = {'train': (s.TrainLoaderConfig, Field(default_factory=dict)),
                                     'valid': (s.EvalLoaderConfig | None, None), 'test': (s.EvalLoaderConfig | None, None)}
                else:
                    loader_fields = {mode: (s.EvalLoaderConfig, Field(default_factory=dict))}
                loaders = make_fields_model(prefix+'Loaders', s.ConfigBase, loader_fields)
                data = make_fields_model(prefix+'Data', s.ConfigBase, {
                    'dataset': (dataset, Field()), 'loaders': (loaders, Field(default_factory=dict)),
                    'collator': (collator, Field()),
                })
                fields['data'] = (data, Field())
            if mode in ('train', 'test'):
                metric_fields = ({split: (list[types['metric']], Field(default_factory=list)) for split in ('train', 'valid', 'test')}
                                 if mode == 'train' else {'test': (list[types['metric']], Field(min_length=1))})
                metrics = make_fields_model(prefix+'Metrics', s.ConfigBase, metric_fields)
                evaluation = make_fields_model(prefix+'Evaluation', s.TrainEvaluationCommonConfig if mode == 'train' else s.ConfigBase,
                    {'metrics': (metrics, Field(default_factory=dict) if mode == 'train' else Field())})
                fields['evaluation'] = (evaluation, Field(default_factory=dict) if mode == 'train' else Field())
            if mode == 'train':
                optimization = make_fields_model(prefix+'Optimization', s.ConfigBase, {
                    'optimizer': (types['optimizer'], Field()), 'scheduler': (Union[types['scheduler'], None], None),
                    'loop': (s.LoopConfig, Field(default_factory=dict)), 'validation': (s.ValidationConfig, Field(default_factory=dict)),
                })
                fields['optimization'] = (optimization, Field())
            elif mode == 'inference':
                inference = make_fields_model(prefix+'Inference', spec['inference_common_schema'], {'decoding': (types['decoding'], Field())})
                fields['inference'] = (inference, Field())
            elif mode == 'visualize':
                visualize = make_fields_model(prefix+'Visualize', s.ConfigBase, {'methods': (list[types['visualizer']], Field(min_length=1))})
                fields['visualize'] = (visualize, Field())
            roots[(task_key, mode)] = make_fields_model(prefix+'Config', s.RootCommonConfig, fields)
    return dict(runtime_registries=registries, component_specs=specs, category_types=types,
                task_specs=tasks, root_schemas=roots, union_routes=union_routes)


def select_root_schema(raw_config: dict, catalog: dict, expected_mode=None) -> tuple:
    """
    :param raw_config: dict, 合并并解析插值后的普通配置树，需要包含 task 和 run 配置块
    :param catalog: dict, build_schema_catalog 返回的规则集合
    :param expected_mode: str | None, 调用入口要求的模式，默认为 None 表示不额外限制入口模式

    :return: tuple[type[ConfigBase], str, str], 按顺序返回根配置类、任务规范名和规范化的运行模式

    该函数用于根据 task.type 与 run.mode 选择根配置类，并核对 expected_mode，不修改输入配置树
    """
    if not isinstance(raw_config, dict):
        raise config_error('配置根必须为 dict。')
    for key in ('task', 'run'):
        if not isinstance(raw_config.get(key), dict):
            raise config_error(f'{key} 必须为配置字典。', (key,))
    task = resolve_task_key(raw_config['task'].get('type'), catalog['task_specs'])
    mode = raw_config['run'].get('mode')
    if not isinstance(mode, str) or mode.strip().casefold() not in MODES:
        raise config_error(f'run.mode 必须为 {MODES} 中的字符串。', ('run', 'mode'))
    mode = mode.strip().casefold()
    if expected_mode is not None:
        if not isinstance(expected_mode, str) or expected_mode.strip().casefold() not in MODES:
            raise config_error(f'入口 expected_mode 不受支持：{expected_mode!r}', ('run', 'mode'))
        if expected_mode.strip().casefold() != mode:
            raise config_error(f'入口要求 {expected_mode!r}，配置却选择 {mode!r}。', ('run', 'mode'))
    if (task, mode) not in catalog['root_schemas']:
        raise config_error(f'未登记任务/模式组合 {(task, mode)}。', ('run', 'mode'))
    return catalog['root_schemas'][(task, mode)], task, mode
