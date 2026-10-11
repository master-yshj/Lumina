
from copy import deepcopy

from loaders.qa import QADataset, QACollator
from tasks.builder.transformer_builder import TransformerBuilder
from tasks.QA_SFT.qa import QATask
from tasks.QA_SFT.inference import QAInference


class QABuilder(TransformerBuilder):
    """
    :param config: dict, Parser 返回的完整配置，传入继承的 BaseBuilder.__init__ 后深拷贝为 self.config。
    :param device: torch.device, 入口选定的运行设备，以关键字参数传入并保存为 self.device。
    :param device_summary: dict, 入口提供的设备信息，必须包含 backend，深拷贝后保存为 self.device_summary。
    :param output_paths: dict, 入口提供的输出路径，metric_files 保存划分名到 CSV Path 的映射，checkpoints_dir 为训练目录 Path 或非训练时的 None。
    :param logger: RunLogger | None, 入口提供的日志对象，默认 None，各组件共享该对象。
    :param runtime_context: RuntimeContext | None, 当前进程的设备和通信上下文，默认 None 时基类使用 device 创建单进程上下文。

    :return: QABuilder, QA 任务组装器实例，调用 build 后返回当前运行模式需要的对象字典。

    该类继承 TransformerBuilder，复用其模型构造和 BaseBuilder 的四种流程，仅实现 QADataset、QACollator、QATask 和 QAInference 的创建；数据窗口上限取自模型配置，推理窗口重叠参数取自 inference.doc_stride。
    """
    def build_dataset(self, split):
        """
        :param self: QABuilder, 当前任务组装器，持有解析后的配置以及公共流程已创建的模型和分词器。
        :param split: str, 当前需要构造的 train、valid、test 或 visualize 划分名。

        :return: QADataset, 使用当前分词器构造的数据集实例，由 build_dataloader 缓存到 self.datasets[split]。

        该方法从 data.dataset.splits[split] 取数据路径，使用当前分词器、日志和 dataset.params 副本构造 QA 数据集，窗口上限取 components.model.params.max_position_embeddings，读取及构建进度间隔固定为 10000。
        """
        spec = self.config['data']['dataset']
        return QADataset(data_file_path=spec['splits'][split], tokenizer=self.tokenizer,
                         split_name=split, progress_interval=10000, read_interval=10000,
                         logger=self.logger,
                         max_position_embeddings=self.config['components']['model']['params']['max_position_embeddings'],
                         **deepcopy(spec['params']))

    def build_collator(self, split):
        """
        :param self: QABuilder, 当前任务组装器，持有解析后的配置以及公共流程已创建的模型和分词器。
        :param split: str, 调用方传入的划分名，当前实现不读取此参数，各划分使用相同配置。

        :return: QACollator, 新建的批次整理对象，由 build_dataloader 缓存到 self.collators[split]。

        该方法读取 data.collator.params 的副本并传入共享的 self.tokenizer，创建负责 Padding、答案位置和元数据整理的 QACollator，每次调用返回独立对象。
        """
        return QACollator(tokenizer=self.tokenizer, **deepcopy(self.config['data']['collator']['params']))

    def build_task(self, loss_fn):
        """
        :param self: QABuilder, 当前任务组装器，持有解析后的配置以及公共流程已创建的模型和分词器。
        :param loss_fn: WeightedLoss | None, 公共流程创建的组合损失，推理和可视化时传入 None。

        :return: QATask, 持有当前 self.model 和传入 loss_fn 的任务实例，由模式组装方法保存到 self.task。

        该方法把已构造模型和损失对象按引用传入任务构造器，任务负责模型前向及对应的损失组织，不重复创建模型。
        """
        return QATask(model=self.model, loss_fn=loss_fn)

    def build_inference_adapter(self, decoding_strategy):
        """
        :param self: QABuilder, 当前任务组装器，持有解析后的配置以及公共流程已创建的模型和分词器。
        :param decoding_strategy: object, BaseBuilder.build_decoding 创建的解码策略实例。

        :return: QAInference, 共享当前模型、分词器和解码策略的推理适配器。

        该方法将共享模型、分词器和解码策略传给 QAInference，窗口容量上限取模型的 max_position_embeddings，窗口重叠 Token 数取 inference.doc_stride，不创建 DataLoader。
        """
        return QAInference(model=self.model, tokenizer=self.tokenizer,
                           decoding_strategy=decoding_strategy,
                           max_position_embeddings=self.config['components']['model']['params']['max_position_embeddings'],
                           doc_stride=self.config['inference']['doc_stride'])
