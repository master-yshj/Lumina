from copy import deepcopy

from registry.base import Registry
from parse import schemas as s
from tasks.builder.base_builder import BaseBuilder
from tasks.builder.errors import BuildError
from tasks.MLM_PreTrain.builder import MLMBuilder
from tasks.QA_SFT.builder import QABuilder


class TaskRegistry(Registry):

    def __init__(self):
        self.category = 'task_builder'
        self.canonical_map = {}
        self.lookup_map = {}

    def register(self, canonical_key, aliases, builder_class, *, config_spec):
        if not isinstance(aliases, (list, tuple)):
            raise TypeError('aliases 必须为 list 或 tuple，请人工检查任务登记。')
        if not isinstance(builder_class, type) or not issubclass(builder_class, BaseBuilder):
            raise TypeError('builder_class 必须为 BaseBuilder 的子类，请人工检查任务登记。')
        if not isinstance(config_spec, dict):
            raise TypeError('config_spec 必须为 dict，请人工检查任务登记。')

        required = {
            'dataset_types', 'dataset_params_schema', 'collator_key',
            'collator_params_schema', 'inference_common_schema',
            'allowed_tokenizers', 'allowed_models', 'allowed_metrics',
            'allowed_decodings', 'modes',
        }
        missing = required - set(config_spec)
        extra = set(config_spec) - required
        if missing or extra:
            raise ValueError(f'任务 {canonical_key!r} 的规则字段不完整：缺少 {missing}，多余 {extra}。')

        spec = deepcopy(config_spec)
        for name in ('dataset_params_schema', 'collator_params_schema', 'inference_common_schema'):
            schema = spec[name]
            if not isinstance(schema, type) or not issubclass(schema, s.ConfigBase):
                raise TypeError(f'任务 {canonical_key!r} 的 {name} 必须为 ConfigBase 子类。')

        for name in ('dataset_types', 'modes'):
            values = spec[name]
            if not isinstance(values, tuple) or not values:
                raise TypeError(f'任务 {canonical_key!r} 的 {name} 必须为非空 tuple。')
            if any(not isinstance(value, str) or not value or value != value.strip().casefold() for value in values):
                raise ValueError(f'任务 {canonical_key!r} 的 {name} 必须只包含规范名称。')
            if len(values) != len(set(values)):
                raise ValueError(f'任务 {canonical_key!r} 的 {name} 不允许重复名称。')
        if not set(spec['modes']) <= {'train', 'test', 'inference', 'visualize'}:
            raise ValueError(f'任务 {canonical_key!r} 包含当前入口不支持的运行模式。')

        collator_key = spec['collator_key']
        if not isinstance(collator_key, str) or not collator_key or collator_key != collator_key.strip().casefold():
            raise ValueError(f'任务 {canonical_key!r} 的 collator_key 必须为规范名称。')

        for name in ('allowed_tokenizers', 'allowed_models', 'allowed_metrics', 'allowed_decodings'):
            values = spec[name]
            if not isinstance(values, (set, frozenset)):
                raise TypeError(f'任务 {canonical_key!r} 的 {name} 必须为 set 或 frozenset。')
            if any(not isinstance(value, str) or not value or value != value.strip().casefold() for value in values):
                raise ValueError(f'任务 {canonical_key!r} 的 {name} 必须只包含规范名称。')

        record = {'builder_class': builder_class, 'config_spec': spec}
        super().register(canonical_key, aliases, record)

    def get(self, canonical_key):
        return super().get(canonical_key)['builder_class']

    def get_spec(self, canonical_key):
        return deepcopy(super().get(canonical_key)['config_spec'])


TASK_BUILDER_REGISTRY = TaskRegistry()

TASK_BUILDER_REGISTRY.register(
    'mlmpretrain', ['mlm_pretrain'], MLMBuilder,
    config_spec={
        'dataset_types': ('demo_mlm',),
        'dataset_params_schema': s.MLMDatasetParamsConfig,
        'collator_key': 'mlm_collator',
        'collator_params_schema': s.MLMCollatorParamsConfig,
        'inference_common_schema': s.MLMInferenceCommonConfig,
        'allowed_tokenizers': {'exampletokenizer'},
        'allowed_models': {'tinybertformaskedlm'},
        'allowed_metrics': {'tokenaccuracy'},
        'allowed_decodings': {'topksampling'},
        'modes': ('train', 'test', 'inference', 'visualize'),
    },
)

TASK_BUILDER_REGISTRY.register(
    'qasft', ['qa_sft'], QABuilder,
    config_spec={
        'dataset_types': ('demo_qa',),
        'dataset_params_schema': s.EmptyParamsConfig,
        'collator_key': 'qa_collator',
        'collator_params_schema': s.EmptyParamsConfig,
        'inference_common_schema': s.QAInferenceCommonConfig,
        'allowed_tokenizers': {'qatokenizer'},
        'allowed_models': {'tinybertforquestionanswering'},
        'allowed_metrics': {'exactmatch', 'characterf1'},
        'allowed_decodings': {'qaspantopkdecoding'},
        'modes': ('train', 'test', 'inference', 'visualize'),
    },
)


def create_builder(config, *, device, device_summary, output_paths, logger=None, runtime_context=None):
    try:
        builder_class = TASK_BUILDER_REGISTRY.get(config['task']['type'])
        return builder_class(config, device=device, device_summary=device_summary,
                             output_paths=output_paths, logger=logger, runtime_context=runtime_context)
    except Exception as error:
        raise BuildError(f"选择 Builder 失败，field=task.type，task={config.get('task')}: {error}") from error
