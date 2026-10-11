
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
from metrics.registry import METRIC_REGISTRY
from metrics.core.collection import MetricCollection
from metrics.output.writer import MetricWriter
from Inference.decoding.registry import DECODING_REGISTRY
from visualize.registry import VISUALIZER_REGISTRY
from Trainer.trainer import Trainer
from Trainer.checkpoint import load_checkpoint, load_checkpoint_state
from tasks.builder.errors import BuildError
from tasks.builder.support import build_metric_fieldnames, visualizer_output_name
from runtime.distributed import RuntimeContext
from torch.utils.data.distributed import DistributedSampler
from run_support import seed_worker
from Trainer.precision import PrecisionController
from torch.nn.parallel import DistributedDataParallel

class BaseBuilder:
    """
    :param config: dict, Parser 返回的完整配置，传入继承的 BaseBuilder.__init__ 后深拷贝为 self.config。
    :param device: torch.device, 入口选定的运行设备，以关键字参数传入并保存为 self.device。
    :param device_summary: dict, 入口提供的设备信息，必须包含 backend，深拷贝后保存为 self.device_summary。
    :param output_paths: dict, 入口提供的输出路径，metric_files 保存划分名到 CSV Path 的映射，checkpoints_dir 为训练目录 Path 或非训练时的 None。
    :param logger: RunLogger | None, 入口提供的日志对象，默认 None，各组件共享该对象。
    :param runtime_context: RuntimeContext | None, 当前进程的设备和通信上下文，默认 None 时基类使用 device 创建单进程上下文。

    :return: BaseBuilder, 保存配置与运行依赖的组装器实例，具体任务应通过子类提供数据和 Task 的构造方法。

    该类统一组织 train、test、inference、visualize 的组件创建顺序；构造时保存状态，调用 build 后才创建运行对象，每个实例只允许组装一次。
    """

    def __init__(self, config, *, device, device_summary, output_paths, logger=None, runtime_context=None):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param config: dict, Parser 返回的完整配置，传入继承的 BaseBuilder.__init__ 后深拷贝为 self.config。
        :param device: torch.device, 入口选定的运行设备，以关键字参数传入并保存为 self.device。
        :param device_summary: dict, 入口提供的设备信息，必须包含 backend，深拷贝后保存为 self.device_summary。
        :param output_paths: dict, 入口提供的输出路径，metric_files 保存划分名到 CSV Path 的映射，checkpoints_dir 为训练目录 Path 或非训练时的 None。
        :param logger: RunLogger | None, 入口提供的日志对象，默认 None，各组件共享该对象。
        :param runtime_context: RuntimeContext | None, 当前进程的设备和通信上下文，默认 None 时基类使用 device 创建单进程上下文。

        :return: None, 初始化 self.config、device、device_summary、output_paths、logger、runtime_context，以及 tokenizer、model、task 和各划分的对象缓存。

        该类初始化检查入口参数的基本结构，深拷贝配置和设备摘要，浅拷贝输出路径字典，共享设备及日志对象；组件初值为 None 或空字典，_build_started 初值为 False。
        """
        if not isinstance(config, dict):
            raise TypeError(f'config 必须为 Parser 返回的普通 dict 喵! 现在的type: {type(config)}')
        for key in ('version', 'task', 'run', 'runtime', 'components'):
            if key not in config:
                raise ValueError(f'config 缺少 Builder 必需字段：{key} 喵!')
        if not isinstance(device, torch.device):
            raise TypeError(f'device 必须为 torch.device 对象喵! 现在的type: {type(device)}')
        if not isinstance(device_summary, dict) or 'backend' not in device_summary:
            raise ValueError(f'device_summary: {device_summary} 必须为包含 backend 的 dict 喵! 现在的type: {type(device_summary)}')
        if not isinstance(output_paths, dict) or not {'metric_files', 'checkpoints_dir'} <= output_paths.keys():
            raise ValueError(f'output_paths: {output_paths} 必须为包含 metric_files/checkpoints_dir 的 dict 喵! 现在的type: {type(output_paths)}')
        if not isinstance(output_paths['metric_files'], dict):
            raise TypeError(f'output_paths.metric_files 必须为 dict[str, Path] 喵! 现在的type: {type(output_paths.metric_files)}')
        self.config = deepcopy(config)
        self.device = device
        if runtime_context is None:
            runtime_context = RuntimeContext(device=device)
        else:
            if not isinstance(runtime_context, RuntimeContext):
                raise TypeError(f"runtime_context: {runtime_context} 必须为RuntimeContext类型的喵! 现在的type: {type(runtime_context)}")
        self.runtime_context = runtime_context
        self.device_summary = deepcopy(device_summary)
        self.output_paths = dict(output_paths)
        self.output_paths['metric_files'] = dict(output_paths['metric_files'])
        self.logger = logger
        self.tokenizer = None
        self.model = None
        self.task = None
        self.datasets = {}
        self.collators = {}
        self.dataloaders = {}
        self.metrics = {}
        self.writers = {}
        self._build_started = False

    def _call(self, action, field, function, *args, **kwargs):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param action: str, 当前组装操作的名称，例如 build_model，用于失败信息。
        :param field: str, 与当前操作有关的配置位置或输出位置，例如 components.model，只用于错误说明。
        :param function: Callable, 调用方传入的函数、绑定方法或类，本方法负责实际调用它。
        :param args: tuple, 传给 function 的位置参数，由调用 _call 时的额外位置参数收集而来。
        :param kwargs: dict, 传给 function 的关键字参数，由调用 _call 时的额外关键字参数收集而来。

        :return: object, function(*args, **kwargs) 的原始返回值，可能是组件实例、字典或 None。

        该方法执行传入的可调用对象；已是 BuildError 的异常原样传播，其他异常补充任务、模式、操作及字段位置，并保留原始异常原因。
        """
        try:
            return function(*args, **kwargs)

        except BuildError:
            raise

        except Exception as error:
            task = self.config['task']['type']
            mode = self.config['run']['mode']

            raise BuildError(f'task={task}, mode={mode}, action={action}, field={field}: {error}') from error

    def _validate_request(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: None, 当前运行请求与入口依赖检查通过后结束，不构造组件或创建输出目录。

        该方法读取 self.config、runtime_context、device_summary 和 output_paths，核对配置版本、模式、精度、通信及设备描述；主进程检查各划分 CSV 路径，所有进程检查 checkpoint 目录要求。
        """
        version = self.config['version']
        if type(version) is not int or version != 2:
            raise ValueError(
                "Builder 只接收新版 Parser 的 version=2 配置喵! "
                f"当前 version: {version!r}"
            )

        mode = self.config['run']['mode']
        if mode not in ('train', 'test', 'inference', 'visualize'):
            raise ValueError(f"未知运行模式喵! 当前 run.mode: {mode!r}")

        runtime = self.config['runtime']
        context = self.runtime_context

        strategy = runtime['distributed']['strategy']
        precision = runtime['precision']

        if mode == 'train':
            if precision not in ('fp32', 'fp16', 'bf16'):
                raise ValueError(
                    "训练模式的 runtime.precision 必须为 fp32、fp16 或 bf16 喵! "
                    f"当前值: {precision!r}"
                )

            if strategy == 'ddp':
                if not context.enabled:
                    raise ValueError(
                        "runtime.distributed.strategy 为 'ddp'，"
                        "但 runtime_context.enabled 为 False 喵!"
                    )

                if context.backend != 'nccl':
                    raise ValueError(
                        "正式 DDP 训练要求通信 backend 为 'nccl' 喵! "
                        f"当前 context.backend: {context.backend!r}"
                    )

                if context.world_size < 2:
                    raise ValueError(
                        "DDP 训练至少需要两个进程喵! "
                        f"当前 context.world_size: {context.world_size}"
                    )

                if runtime['device']['type'] != 'cuda':
                    raise ValueError(
                        "正式 DDP 训练要求 runtime.device.type 为 'cuda' 喵! "
                        f"当前值: {runtime['device']['type']!r}"
                    )

                if runtime['device']['index'] is not None:
                    raise ValueError(
                        "DDP 训练要求 runtime.device.index 为 None，"
                        "由各进程的 LOCAL_RANK 决定设备编号喵! "
                        f"当前值: {runtime['device']['index']!r}"
                    )

            elif strategy == 'none':
                if context.enabled:
                    raise ValueError(
                        "runtime.distributed.strategy 为 'none'，"
                        "但 runtime_context.enabled 为 True 喵!"
                    )

                if context.world_size != 1:
                    raise ValueError(
                        "strategy='none' 时，context.world_size 必须为 1 喵! "
                        f"当前值: {context.world_size}"
                    )

            else:
                raise ValueError(
                    "训练模式的 runtime.distributed.strategy "
                    "必须为 'none' 或 'ddp' 喵! "
                    f"当前值: {strategy!r}"
                )

        else:
            if precision != 'fp32':
                raise NotImplementedError(
                    f"独立 {mode} 模式目前只支持 precision='fp32' 喵! "
                    f"当前值: {precision!r}"
                )

            if strategy != 'none':
                raise NotImplementedError(
                    f"独立 {mode} 模式目前只支持 strategy='none' 喵! "
                    f"当前值: {strategy!r}"
                )

            if context.enabled:
                raise ValueError(
                    f"独立 {mode} 模式要求 runtime_context.enabled 为 False，"
                    "不能使用已启用通信的运行上下文喵!"
                )

        if not isinstance(self.device, torch.device):
            raise TypeError('入口必须提供 torch.device 对象喵!')
        if not self.device==context.device:
            raise ValueError(f"当前选择设备: {self.device} 与context设备: {context.device} 不一致喵!")
        requested = runtime['device']
        if context.device.type != requested['type']:
            raise ValueError(f'传入 device: {requested["type"]} 与 context.device.type: {context.device.type} 不一致')
        if strategy == 'ddp':
            if not context.device.index==context.local_rank:
                raise ValueError(f"context.device.index: {context.device.index} 与 context.local_rank: {context.local_rank} 不一致喵!")
        elif strategy == 'none' and requested['index'] is not None:
            if not requested['index'] == context.device.index:
                raise ValueError(f"requested['index']: {requested['index']} 与 context.device.index: {context.device.index} 不一致喵!")
        if self.device_summary['backend'] != context.device.type:
            raise ValueError(f"device_summary[backend]: {self.device_summary['backend']} 与 context.device.type: {context.device.type} 不一致喵!")
        for key in ('rank','local_rank','world_size'):
            if key in self.device_summary:
                expected_value = getattr(context, key)
                if self.device_summary[key] != expected_value:
                    raise ValueError(f"device_summary[{key}]: {self.device_summary[key]} 与 expected_value: {expected_value} 不一致喵!")
        if context.is_main:
            metric_splits = (
                self.get_active_splits()
                if mode in ('train', 'test')
                else []
            )

            metric_files = self.output_paths['metric_files']
            paths = []

            for split in metric_splits:
                if split not in metric_files:
                    raise ValueError(
                        f"入口没有提供 output_paths.metric_files.{split} 喵!"
                    )

                path = metric_files[split]

                if not isinstance(path, Path):
                    raise TypeError(
                        f"output_paths.metric_files.{split} 必须为 Path 对象喵! "
                        f"当前类型: {type(path)}"
                    )

                if not path.is_absolute():
                    raise ValueError(
                        f"output_paths.metric_files.{split} 必须为绝对路径喵! "
                        f"当前路径: {path}"
                    )

                if path.suffix.casefold() != '.csv':
                    raise ValueError(
                        f"output_paths.metric_files.{split} 必须以 .csv 结尾喵! "
                        f"当前路径: {path}"
                    )

                if not path.parent.is_dir():
                    raise ValueError(
                        f"output_paths.metric_files.{split} 的父目录不存在或不是目录喵! "
                        f"父目录: {path.parent}"
                    )

                if path.exists() and not path.is_file():
                    raise ValueError(
                        f"output_paths.metric_files.{split} 已存在，但不是文件喵! "
                        f"当前路径: {path}"
                    )

                paths.append(path.resolve())

            if len(set(paths)) != len(paths):
                raise ValueError(
                    f"不同划分不能共用同一个 CSV 文件喵! 当前路径列表: {paths}"
                )

        checkpoint_dir = self.output_paths['checkpoints_dir']

        if mode == 'train':
            if (
                    not isinstance(checkpoint_dir, Path)
                    or not checkpoint_dir.is_absolute()
                    or not checkpoint_dir.is_dir()
            ):
                raise ValueError(
                    "训练需要入口提供已经存在的绝对 checkpoints_dir Path 喵! "
                    f"当前值: {checkpoint_dir!r}"
                )

        elif checkpoint_dir is not None:
            raise ValueError(
                "非训练模式的 checkpoints_dir 必须为 None 喵! "
                f"当前值: {checkpoint_dir!r}"
            )

    def get_active_splits(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: list[str], 当前进程需要的数据划分；推理为 []，独立测试为 ["test"]，可视化为 ["visualize"]。

        该方法训练时所有进程包含 train，只有主进程按 validation.enabled 和 test_after_train 决定是否追加 valid、test，供组装流程选择需要创建的 DataLoader。
        """
        mode = self.config['run']['mode']
        if mode == 'train':
            splits = ['train']
            if self.runtime_context.is_main:
                if self.config['optimization']['validation']['enabled']:
                    splits.append('valid')
                if self.config['evaluation']['test_after_train']:
                    splits.append('test')
            return splits
        return {'test': ['test'], 'visualize': ['visualize'], 'inference': []}[mode]

    def _make_result(self, **updates):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param updates: dict, 用关键字参数传入的结果字段及值，只允许覆盖下面列出的固定字段。

        :return: dict, 包含 task、model、tokenizer、dataloaders、metrics、writers、optimizer、scheduler、trainer、fit_kwargs、task_inference、visualizers。

        该方法新建返回字典并浅拷贝三个按划分保存对象的字典，各组件仍共享原对象；未使用字段保留 None、{} 或 []，最后应用 updates 供不同模式补充结果。
        """
        result = dict(task=self.task, model=self.model, tokenizer=self.tokenizer,
                      dataloaders=dict(self.dataloaders), metrics=dict(self.metrics),
                      writers=dict(self.writers), optimizer=None, scheduler=None,
                      trainer=None, fit_kwargs={}, task_inference=None, visualizers=[])
        unknown = updates.keys() - result.keys()
        if unknown:
            raise ValueError(f'未知返回字段：{unknown}')
        result.update(updates)
        return result

    def build_tokenizer(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: object, TOKENIZER_REGISTRY 中对应名称的工厂创建的分词器实例。

        该方法从 self.config["components"]["tokenizer"] 读取 type 和 params，深拷贝参数后调用注册对象；由 _build_common_components 将返回值保存为 self.tokenizer。
        """
        spec = self.config['components']['tokenizer']
        return TOKENIZER_REGISTRY.get(spec['type'])(**deepcopy(spec['params']))

    def build_model(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: torch.nn.Module, MODEL_REGISTRY 中对应名称的工厂创建的模型实例。

        该方法从 self.config["components"]["model"] 读取 type 和 params，深拷贝参数后直接构造模型；需要由分词器派生参数的模型族应覆盖本方法。
        """
        spec = self.config['components']['model']
        return MODEL_REGISTRY.get(spec['type'])(**deepcopy(spec['params']))

    def initialize_model(self, init_config):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param init_config: dict | None, components.model.init_from 的值，非空时描述初始化方式及权重路径。

        :return: None, init_config 为 None 时直接结束，基类不执行权重初始化。

        该方法作为模型族初始化接口；非空初始化在本基类中尚未实现，由 TransformerBuilder 等子类覆盖并加载所需权重。
        """
        if init_config is not None:
            raise NotImplementedError(f'{type(self).__name__} 尚未实现 model.init_from')

    def build_losses(self, loss_configs):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param loss_configs: list[dict], 损失项配置列表，每项含 name、type、weight、params，可来自 components.losses 或损失曲面配置。

        :return: WeightedLoss, 保存各损失函数、名称及权重的组合损失对象。

        该方法按 type 从 LOSS_REGISTRY 取得工厂并用 params 副本构造损失函数，再组成 name、weight、loss_fn 记录交给 WeightedLoss，不修改原配置。
        """
        losses = []
        for spec in loss_configs:
            loss_fn = LOSS_REGISTRY.get(spec['type'])(**deepcopy(spec['params']))
            losses.append(dict(name=spec['name'], weight=spec['weight'], loss_fn=loss_fn))
        return WeightedLoss(losses)

    def build_dataset(self, split):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param split: str, 要构造的数据划分名，例如 train、valid、test 或 visualize。

        :return: Dataset, 子类实现时应返回该划分的数据集；当前基类没有可用返回结果。

        该方法声明具体任务必须实现的数据集构造接口，实际文件路径和参数的取法由 MLMBuilder、QABuilder 等任务子类决定。
        """
        raise NotImplementedError(f'{type(self).__name__} 必须实现 build_dataset({split!r})')

    def build_collator(self, split):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param split: str, 当前数据划分名，由 build_dataloader 传入。

        :return: Callable, 子类应返回把样本列表整理成 batch 的对象；当前基类没有可用返回结果。

        该方法声明具体任务必须实现的批次整理接口，供 DataLoader 通过 collate_fn 调用。
        """
        raise NotImplementedError(f'{type(self).__name__} 必须实现 build_collator({split!r})')

    def build_dataloader(self, split):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param split: str, 当前要构造的划分名，必须存在于 data.loaders 和 data.dataset.splits 中。

        :return: DataLoader, 新建的数据加载器，同时缓存到 self.dataloaders[split]。

        该方法调用子类创建 Dataset 和 Collator，读取该划分的加载参数；DDP 训练使用 DistributedSampler，随机生成器使用 experiment.seed 加 rank，检查非零批次后缓存三个对象并记录样本统计。
        """
        if split in self.dataloaders:
            raise ValueError(f'重复构造数据划分：{split}')
        dataset=self._call('build_dataset',f'data.dataset.splits.{split}',self.build_dataset,split)
        collator=self._call('build_collator','data.collator',self.build_collator,split)
        local_loader_kwargs = deepcopy(self.config['data']['loaders'][split])
        sampler = None
        if split == 'train' and self.runtime_context.enabled:
            sampler = DistributedSampler(dataset, num_replicas=self.runtime_context.world_size,
                                         rank=self.runtime_context.rank, shuffle=local_loader_kwargs['shuffle'],
                                         seed=self.config['experiment']['seed'], drop_last=False)
            del local_loader_kwargs['shuffle']
        generator = torch.Generator()
        generator.manual_seed(self.config['experiment']['seed'] + self.runtime_context.rank)
        if sampler is None:
            loader = DataLoader(dataset, collate_fn=collator, worker_init_fn=seed_worker, generator=generator, **local_loader_kwargs)
        else:
            loader = DataLoader(dataset, collate_fn=collator, sampler=sampler, shuffle=False,
                                worker_init_fn=seed_worker, generator=generator, **local_loader_kwargs)
        if len(loader) == 0:
            raise ValueError(f'{split} 没有可用 batch，请检查数据内容、batch_size 和 drop_last')
        self.datasets[split] = dataset
        self.collators[split] = collator
        self.dataloaders[split] = loader
        loader_description = {
            'dataset_size': len(dataset),
            'loader_batches': len(loader),
            'batch_size': loader.batch_size,
            'loader_drop_last': loader.drop_last
        }
        if sampler is not None:
            loader_description['sampler_size'] = len(sampler)
            loader_description['padding'] = sampler.total_size - len(dataset)
        if self.logger is not None:
            self.logger.info(str(loader_description))
        return loader

    def build_metrics(self, split):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param split: str, 要读取的指标划分，对应 evaluation.metrics 中的键。

        :return: MetricCollection, 当前划分的指标集合，配置空列表时返回空集合对象。

        该方法按 evaluation.metrics[split] 中各项的 type 和 params 副本构造指标，将返回集合交给 _build_evaluation_split 缓存。
        """
        specs = self.config['evaluation']['metrics'][split]
        metrics = [METRIC_REGISTRY.get(s['type'])(**deepcopy(s['params'])) for s in specs]
        return MetricCollection(metrics)

    def build_writer(self, split, metric_collection):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param split: str, 指标所属划分，只允许 train、valid 或 test。
        :param metric_collection: MetricCollection, 当前划分已创建的指标集合，output_fields() 提供其输出列名。

        :return: MetricWriter, 使用 output_paths["metric_files"][split] 和拼好的列名创建的 CSV 写入器。

        该方法从 components.losses 提取损失项名称，与指标输出字段一起生成表头；构造 MetricWriter 时会检查已有 CSV 表头或创建表头，由调用方限制仅在主进程执行。
        """
        loss_names = [s['name'] for s in self.config['components']['losses']]
        fields = build_metric_fieldnames(split, loss_names, metric_collection.output_fields())
        return MetricWriter(metrics_path=self.output_paths['metric_files'][split], fieldnames=fields)

    def build_optimizer(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: torch.optim.Optimizer, 使用当前 self.task.parameters() 构造的优化器实例。

        该方法读取 optimization.optimizer 的 type 与 params，通过 OPTIMIZER_REGISTRY 创建优化器；Task 应先迁移到目标设备，优化器持有同一 Task 的参数引用。
        """
        spec = self.config['optimization']['optimizer']
        return OPTIMIZER_REGISTRY.get(spec['type'])(params=self.task.parameters(), **deepcopy(spec['params']))

    def build_scheduler(self, optimizer):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param optimizer: torch.optim.Optimizer, 本次 build_optimizer 创建的优化器对象。

        :return: object | None, 与传入 optimizer 关联的调度器；optimization.scheduler 为 None 时返回 None。

        该方法读取调度配置并要求 interval 为 step，仅把 optimizer 和 params 副本传给 SCHEDULER_REGISTRY 中的工厂，interval 留给执行流程解释。
        """
        spec = self.config['optimization']['scheduler']
        if spec is None:
            return None
        if spec['interval'] != 'step':
            raise ValueError('当前 Trainer 只支持按优化器更新步推进 Scheduler')
        return SCHEDULER_REGISTRY.get(spec['type'])(optimizer=optimizer, **deepcopy(spec['params']))

    def build_task(self, loss_fn):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param loss_fn: WeightedLoss | None, 训练和测试使用的组合损失，推理和可视化传入 None。

        :return: torch.nn.Module, 子类应返回引用 self.model 的任务对象；当前基类没有可用返回结果。

        该方法声明具体任务的构造接口，让任务子类决定如何调用模型及组织损失，公共流程只负责传入共享模型所需的依赖。
        """
        raise NotImplementedError(f'{type(self).__name__} 必须实现 build_task(loss_fn)')

    def build_inference_adapter(self, decoding_strategy):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param decoding_strategy: object, build_decoding 创建的解码策略实例。

        :return: object, 子类应返回提供 infer 和 read_input 方法的推理适配器；当前基类没有可用返回结果。

        该方法声明任务推理适配器的构造接口，子类应把 self.model、self.tokenizer 和解码策略传入适配器。
        """
        raise NotImplementedError(f'{type(self).__name__} 必须实现 build_inference_adapter')

    def build_decoding(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: object, DECODING_REGISTRY 中对应名称的工厂创建的解码策略实例。

        该方法读取 inference.decoding 的 type 与 params，用参数副本构造策略，再由 build_inference 交给任务推理适配器。
        """
        spec = self.config['inference']['decoding']
        return DECODING_REGISTRY.get(spec['type'])(**deepcopy(spec['params']))

    def build_visualizers(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: list[dict], 每项含 name 输出子目录名和 visualizer 可视化对象，顺序与 visualize.methods 一致。

        该方法逐项读取可视化配置并从注册表创建对象；losslandscape 的 losses 配置先转换为 WeightedLoss，以 loss_fn 参数传入；序号保证同类方法拥有不同输出名称。
        """
        result = []
        for index, spec in enumerate(self.config['visualize']['methods'], 1):
            params = deepcopy(spec['params'])
            if spec['type'] == 'losslandscape':
                params['loss_fn'] = self.build_losses(params.pop('losses'))
            visualizer = VISUALIZER_REGISTRY.get(spec['type'])(**params)
            result.append(dict(name=visualizer_output_name(index, spec['type']), visualizer=visualizer))
        return result

    def build_fit_kwargs(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: dict, Trainer.fit 的关键字参数，包括 loop 字段、训练及验证 DataLoader、验证频率、早停、保存间隔和 checkpoint 目录。

        该方法复制 optimization.loop 并填入已有运行对象及 validation 设置；关闭验证时频率和早停耐心值为 None，缺少验证加载器时传 None，DataLoader 保持原引用。
        """
        valid_dataloader = None
        if 'valid' in self.dataloaders:
            valid_dataloader = self.dataloaders['valid']

        optimization = self.config['optimization']
        validation = optimization['validation']
        enabled = validation['enabled']
        early = validation['early_stopping']

        return dict(
            deepcopy(optimization['loop']),
            train_dataloader=self.dataloaders['train'],
            valid_dataloader=valid_dataloader,
            validation_enabled=enabled,
            validation_every_steps=validation['every_steps'] if enabled else None,
            early_stopping_patience=(
                early['patience'] if enabled and early['enabled'] else None
            ),
            checkpoint_save_every_rounds=validation['checkpoint']['save_every_rounds'],
            checkpoints_dir=self.output_paths['checkpoints_dir'],
        )

    def build_trainer(self, optimizer, scheduler, precision):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param optimizer: torch.optim.Optimizer, 当前 Task 对应的优化器实例。
        :param scheduler: object | None, 关联同一优化器的调度器，未配置时为 None。
        :param precision: PrecisionController, 按 runtime.precision 和当前设备创建的精度控制器。

        :return: Trainer, 持有当前 Task、优化器、调度器、指标、写入器及运行上下文的训练器实例。

        该方法把已创建的对象按引用传给 Trainer；未创建的验证指标或写入器使用 None，训练指标取 self.metrics["train"]，此处不调用 fit。
        """
        train_writer = None
        valid_writer = None
        valid_metrics = None
        if 'train' in self.writers:
            train_writer = self.writers['train']
        if 'valid' in self.writers:
            valid_writer = self.writers['valid']
        if 'valid' in self.metrics:
            valid_metrics = self.metrics['valid']
            train_metrics = self.metrics['train']
        return Trainer(task=self.task,optimizer=optimizer,scheduler=scheduler,device=self.device,device_summary=self.device_summary,train_metric_collection=self.metrics['train'],valid_metric_collection=valid_metrics,train_metrics_writer=train_writer,valid_metrics_writer=valid_writer,logger=self.logger,runtime_context=self.runtime_context,precision=precision)

    def restore_training(self, trainer, optimizer, scheduler):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param trainer: Trainer, 已创建的训练器，加载时恢复其训练进度和精度控制状态等。
        :param optimizer: torch.optim.Optimizer, 当前 Task 的优化器，加载时恢复其状态。
        :param scheduler: object | None, 当前优化器的调度器，是否存在应与 checkpoint 匹配。

        :return: None, 状态恢复到现有 self.task、optimizer、scheduler 和 trainer 对象中。

        该方法读取 run.resume_from；非空时先通过 load_checkpoint 将文件加载到 CPU，再交给 load_checkpoint_state 恢复完整训练状态，为空时直接结束。
        """
        path = self.config['run']['resume_from']
        if path is not None:
            checkpoint = load_checkpoint(path, map_location='cpu')
            load_checkpoint_state(checkpoint, self.task, optimizer, scheduler, trainer)

    def _check_object_graph(self, result):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param result: dict, 当前运行模式由 _make_result 组织出的对象字典。

        :return: None, 所有引用关系检查通过后结束，不创建或替换对象。

        该方法用对象身份检查 Task、模型、返回字典、Trainer、调度器和推理适配器是否共享本次组件；同时核对优化器包含 Task 的完整且不重复的参数集合及训练包装对象。
        """
        if self.task.model is not self.model:
            raise ValueError('Task.model 不是 Builder 创建的模型')
        for name in ('model', 'task', 'tokenizer'):
            if result[name] is not getattr(self, name):
                raise ValueError(f'返回的 {name} 与 Builder 中的对象不同')
        for name in ('dataloaders', 'metrics', 'writers'):
            cache = getattr(self, name)
            if result[name].keys() != cache.keys() or any(result[name][k] is not v for k, v in cache.items()):
                raise ValueError(f'返回的 {name} 未保留原对象引用')
        trainer, optimizer, scheduler = result['trainer'], result['optimizer'], result['scheduler']
        if trainer is not None:
            if trainer.task is not self.task or trainer.optimizer is not optimizer or trainer.scheduler is not scheduler:
                raise ValueError('Trainer 持有不同的 Task/Optimizer/Scheduler')
            if not trainer.runtime_context is self.runtime_context:
                raise ValueError(f"当前trainer上下文: {trainer.runtime_context} 与 runtime_context上下文: {self.runtime_context} 不一致喵!")
            if not trainer.precision.device==self.device:
                raise ValueError(f"当前精度设备: {trainer.precision.device} 与所选设备: {self.device} 不一致喵!")
            if self.runtime_context.enabled:
                if not trainer.train_task.module is trainer.task:
                    raise ValueError(f"启用通信时 train_task.module: {trainer.train_task.module} 必须是 trainer.task: {trainer.task} 类型的喵!")
            else:
                if not trainer.train_task is trainer.task:
                    raise ValueError(f"未启用通信时 train_task: {trainer.train_task} 必须是 trainer.task: {trainer.task} 类型的喵!")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError('Scheduler 未关联本次 Optimizer')
        if optimizer is not None:
            expected = {id(p) for p in self.task.parameters()}
            actual = [id(p) for group in optimizer.param_groups for p in group['params']]
            if set(actual) != expected or len(actual) != len(set(actual)):
                raise ValueError('Optimizer 参数不是当前 Task 的完整且不重复的参数集合')
        adapter = result['task_inference']
        if adapter is not None and adapter.model is not self.model:
            raise ValueError('推理适配器未共享当前模型')

    def _build_evaluation_split(self, split):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param split: str, 要构造的 train、valid 或 test 划分。

        :return: None, 加载器、指标及主进程的写入器分别保存到 self.dataloaders、self.metrics、self.writers 的 split 键。

        该方法为同一划分依次创建 DataLoader 和 MetricCollection，主进程再创建 MetricWriter；训练划分也复用本方法，仅进行组装，不执行训练或评估。
        """
        self._call('build_dataloader', f'data.loaders.{split}', self.build_dataloader, split)
        self.metrics[split]=self._call('build_metrics',f'evaluation.metrics.{split}',self.build_metrics,split)
        if not self.runtime_context.is_main:
            return None
        self.writers[split]=self._call('build_writer',f'output_paths.metric_files.{split}',self.build_writer,split,self.metrics[split])
        return None

    def build_train(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: dict, 包含训练对象及 fit_kwargs 的组装结果，Trainer 已完成所需恢复和训练 Task 包装。

        该方法先通过 run_local 组装各进程的本地训练对象，再恢复状态、比较训练描述并设置 trainer.train_task；只准备训练依赖，不启动 fit。
        """
        context = self.runtime_context
        result = context.run_local('组装本地训练对象', self._build_train_local)
        trainer = result['trainer']
        optimizer = result['optimizer']
        scheduler = result['scheduler']
        fit_kwargs = result['fit_kwargs']
        def restore_local():
            """
            :return: None, 恢复结果写入外层已有的 trainer、optimizer、scheduler 和 self.task 对象。

            该方法交给 context.run_local 在各进程执行，读取外层 trainer、optimizer、scheduler 并调用 self.restore_training；run.resume_from 非空时加载训练状态，为空时不恢复。
            """
            self._call('restore_training', 'run.resume_from', self.restore_training, trainer, optimizer, scheduler)
            return None
        context.run_local('恢复训练状态', restore_local)
        self._check_distributed_training_inputs(trainer, fit_kwargs)
        self.wrap_training_task(trainer)
        return result

    def build_test(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: dict, 包含 Task、模型、分词器及 test 加载器、指标和写入器的组装结果。

        该方法用 components.losses 创建组合损失与 Task，再构造 test 划分；checkpoint 加载和测试执行由测试 Runner 完成。
        """
        loss_fn = self._call('build_losses', 'components.losses', self.build_losses, self.config['components']['losses'])
        self.task = self._call('build_task', 'task.type', self.build_task, loss_fn)
        self._build_evaluation_split('test')
        return self._make_result()

    def build_inference(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: dict, 包含 Task、模型、分词器和 task_inference 适配器的组装结果，数据加载器与指标字典为空。

        该方法创建不带损失的 Task、解码策略和任务推理适配器，共享当前模型与分词器；推理输入由适配器处理，checkpoint 加载及推理由推理 Runner 完成。
        """
        self.task = self._call('build_task', 'task.type', self.build_task, None)
        strategy = self._call('build_decoding', 'inference.decoding', self.build_decoding)
        adapter = self._call('build_inference_adapter', 'inference', self.build_inference_adapter, strategy)
        return self._make_result(task_inference=adapter)

    def build_visualize(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: dict, 包含 Task、模型、分词器、visualize 加载器和 visualizers 方法列表的组装结果。

        该方法创建不带损失的 Task、可视化数据加载器及方法对象，返回后由可视化入口加载 checkpoint 并调用各方法。
        """
        self.task = self._call('build_task', 'task.type', self.build_task, None)
        self._call('build_dataloader', 'data.loaders.visualize', self.build_dataloader, 'visualize')
        visualizers = self._call('build_visualizers', 'visualize.methods', self.build_visualizers)
        return self._make_result(visualizers=visualizers)

    def build(self):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。

        :return: dict, run.mode 对应的完整组装结果，字段结构由 _make_result 固定。

        该方法先协调公共组件构造，再按 run.mode 调用训练、测试、推理或可视化组装方法，检查对象引用后返回；开始过组装的实例不能再次使用。
        """
        self.runtime_context.run_local('构造公共组件',self._build_common_components,)

        dispatch = {
            'train': self.build_train,
            'test': self.build_test,
            'inference': self.build_inference,
            'visualize': self.build_visualize,
        }

        result = self._call(
            'build_mode',
            'run.mode',
            dispatch[self.config['run']['mode']],
        )
        def check_result():
            """
            :return: None, 对象引用检查通过后结束，不创建或替换运行对象。

            该方法读取外层 self 和 result，调用 self._check_object_graph 检查返回字典与 Builder 是否引用同一批对象；由 runtime_context.run_local 协调各进程的检查结果。
            """
            self._call('check_object_graph', 'result', self._check_object_graph, result)
            return None
        self.runtime_context.run_local('检查对象引用', check_result)
        self._call('check_object_graph', 'result', self._check_object_graph, result)
        return result

    def _build_common_components(self):
        """
        :param self: BaseBuilder, 当前组装器实例，持有解析后的配置、运行上下文及待填充的组件属性。

        :return: None, 新建的分词器和模型分别保存到 self.tokenizer 与 self.model。

        该方法由 build 通过 runtime_context.run_local 调用；先设置 _build_started 防止重复组装，再检查请求，依次构造分词器和模型，发生失败后也不重置开始标记。
        """
        if self._build_started:
            raise BuildError(f"build已经开始了喵! build_started: {self._build_started}")
        else:
            self._build_started = True
        self._call('validate_request', 'runtime/output_paths', self._validate_request)
        self.tokenizer = self._call('build_tokenizer', 'components.tokenizer', self.build_tokenizer)
        self.model = self._call('build_model', 'components.model', self.build_model)
        return None

    def _build_train_local(self):
        """
        :param self: BaseBuilder, 已由 _build_common_components 创建分词器和模型的当前实例。

        :return: dict, _make_result 返回的组装字典，其中 optimizer、scheduler、trainer 和 fit_kwargs 保存本次创建的对象及训练调用参数。

        该方法读取 self.config，依次初始化模型权重、创建损失与 Task、迁移 Task 到设备、构造本进程启用的数据划分和指标，再创建优化器、调度器、精度控制器、Trainer；恢复训练和 DDP 包装由外层 build_train 执行。
        """
        init_config = self.config['components']['model']['init_from']
        self._call('initialize_model', 'components.model.init_from', self.initialize_model, init_config)
        loss_fn = self._call('build_losses', 'components.losses', self.build_losses, self.config['components']['losses'])
        self.task = self._call('build_task', 'task.type', self.build_task, loss_fn)
        moved_task = self._call('move_task', 'runtime.device', self.task.to, self.device)
        if moved_task is self.task:
            self.task = moved_task
        for split in self.get_active_splits():
            self._build_evaluation_split(split)
        optimizer = self._call('build_optimizer', 'optimization.optimizer', self.build_optimizer)
        scheduler = self._call('build_scheduler', 'optimization.scheduler', self.build_scheduler,optimizer)
        precision = self._call('build_precision', 'runtime.precision', PrecisionController, self.runtime_context.device, self.config['runtime']['precision'])
        trainer = self._call('build_trainer', 'optimization', self.build_trainer, optimizer, scheduler, precision)
        fit_kwargs = self._call('build_fit_kwargs', 'optimization', self.build_fit_kwargs)
        result = self._make_result(optimizer=optimizer, scheduler=scheduler, trainer=trainer, fit_kwargs=fit_kwargs)
        return result

    def _check_distributed_training_inputs(self,trainer,fit_kwargs):
        """
        :param self: BaseBuilder, 已构造训练 DataLoader、指标和 Task 的当前实例。
        :param trainer: Trainer, 本次组装且已执行恢复步骤的训练器，用于读取任务参数、精度和训练进度。
        :param fit_kwargs: dict, build_fit_kwargs 返回的 Trainer.fit 关键字参数，含累积步数、训练目标、日志及验证设置。

        :return: None, 检查通过后调用 self.metrics["train"].prepare_distributed，为训练指标准备当前通信上下文。

        该方法先通过 run_local 生成本进程训练描述；启用通信时用 torch.distributed.all_gather_object 收集各进程描述并逐项与 rank 0 比较，确认训练循环和参数结构一致，此处不比较模型权重数值。
        """
        context = self.runtime_context
        def describe_inputs():
            """
            :return: dict, 本进程的批次数、批大小、丢尾设置、循环参数、精度、训练进度、数据轮次、损失名称及权重，以及参数名称、形状和 requires_grad。

            该方法读取外层 self.dataloaders["train"]、self.config、trainer、fit_kwargs 和 context，拒绝零批次的训练加载器，整理出可供各进程比较的普通数据；不取出 batch，也不执行训练。
            """
            train_loader = self.dataloaders['train']
            if not len(train_loader)>0:
                raise ValueError(f"train_dataloader: {train_loader} 不能为空喵!")
            if context.enabled:
                buffer_names = [name for name, _ in trainer.task.named_buffers()]
            loop_keys = ('accumulation_steps', 'total_steps', 'log_every_steps', 'validation_enabled',
                         'validation_every_steps', 'early_stopping_patience', 'checkpoint_save_every_rounds')
            loop_options = {key: fit_kwargs[key] for key in loop_keys}
            description = {
                'loader_batches': len(train_loader),
                'batch_size': train_loader.batch_size,
                'drop_last': train_loader.drop_last,
                'loop_options': loop_options,
                'precision': trainer.precision.precision,
                'progress': trainer.get_training_progress(),
                'data_epoch': trainer.data_epoch,
                'losses': tuple((spec['name'],spec['weight']) for spec in self.config['components']['losses']),
                'parameters': tuple((name,tuple(p.shape),p.requires_grad) for name,p in sorted(trainer.task.named_parameters())),
            }
            return description

        description = context.run_local('检查训练对象描述', describe_inputs)
        if context.enabled:
            descriptions = [None] * context.world_size
            dist.all_gather_object(descriptions, description)
            reference = descriptions[0]
            for rank, rank_description in enumerate(descriptions):
                missing_keys = set(reference) - set(rank_description)
                extra_keys = set(rank_description) - set(reference)

                if missing_keys or extra_keys:
                    raise ValueError(
                        f"训练对象描述的字段不一致喵! rank {rank} 相对于 rank 0：\n"
                        f"缺失字段: {sorted(missing_keys)}\n"
                        f"多余字段: {sorted(extra_keys)}"
                    )

                for key in sorted(reference):
                    reference_value = reference[key]
                    rank_value = rank_description[key]

                    if rank_value != reference_value:
                        raise ValueError(
                            f"训练对象描述不一致喵! 字段: {key!r}\n"
                            f"rank 0 的值: {reference_value!r}\n"
                            f"rank {rank} 的值: {rank_value!r}"
                        )
        self.metrics['train'].prepare_distributed(context)
        return None

    def wrap_training_task(self, trainer):
        """
        :param self: BaseBuilder, 当前组装器实例，配置保存在 self.config，已创建的组件保存在对应实例属性中。
        :param trainer: Trainer, 当前训练器，trainer.task 是已构造并迁移到设备的原始 Task。

        :return: None, 结果保存到 trainer.train_task，trainer.task 仍保留原始 Task。

        该方法单进程令 train_task 与 task 指向同一对象；启用通信时用 PyTorch DistributedDataParallel 包装 task，CUDA 使用 local_rank 指定设备，CPU 分支不传 CUDA 参数。
        """
        context = self.runtime_context

        if not context.enabled:
            trainer.train_task = trainer.task
            return None

        if context.device.type == 'cuda':
            wrapped_task = DistributedDataParallel(
                trainer.task,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            wrapped_task = DistributedDataParallel(
                trainer.task,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        trainer.train_task = wrapped_task
        return None
