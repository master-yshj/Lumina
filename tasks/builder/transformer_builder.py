
from copy import deepcopy

from models.Transformer.modeling.registry import MODEL_REGISTRY
from Trainer.checkpoint import load_checkpoint, get_task_state
from tasks.builder.base_builder import BaseBuilder


class TransformerBuilder(BaseBuilder):
    """
    :param config: dict, Parser 返回的完整配置，传入继承的 BaseBuilder.__init__ 后深拷贝为 self.config。
    :param device: torch.device, 入口选定的运行设备，以关键字参数传入并保存为 self.device。
    :param device_summary: dict, 入口提供的设备信息，必须包含 backend，深拷贝后保存为 self.device_summary。
    :param output_paths: dict, 入口提供的输出路径，metric_files 保存划分名到 CSV Path 的映射，checkpoints_dir 为训练目录 Path 或非训练时的 None。
    :param logger: RunLogger | None, 入口提供的日志对象，默认 None，各组件共享该对象。
    :param runtime_context: RuntimeContext | None, 当前进程的设备和通信上下文，默认 None 时基类使用 device 创建单进程上下文。

    :return: TransformerBuilder, 保存配置和入口依赖的组装器实例，调用 build 时才创建运行组件。

    该类继承 BaseBuilder 的构造参数和四种组装流程，补充 Transformer 模型族的参数派生与主干权重初始化；数据、Task 和推理适配器由具体任务子类实现。
    """
    def build_model(self):
        """
        :param self: TransformerBuilder, 已创建 self.tokenizer 且持有 components.model 配置的当前组装器。

        :return: torch.nn.Module, 按模型注册名称创建的新模型，由公共组装流程保存到 self.model。

        该方法深拷贝 model.params 并移除仅保留作元数据的 model_path，从 len(self.tokenizer) 和 pad_token_id 派生 vocab_size、padding_idx，再交给 MODEL_REGISTRY 中的模型工厂。
        """
        spec = self.config['components']['model']
        params = deepcopy(spec['params'])
        params.pop('model_path', None)
        for name, value in dict(vocab_size=len(self.tokenizer), padding_idx=self.tokenizer.pad_token_id).items():
            if name in params:
                raise ValueError(f'components.model.params.{name} 由 Tokenizer 派生，不能重复配置')
            params[name] = value
        return MODEL_REGISTRY.get(spec['type'])(**params)

    def initialize_model(self, init_config):
        """
        :param self: TransformerBuilder, 已创建 self.model 的当前组装器，非空初始化要求该模型具有 bert 属性。
        :param init_config: dict | None, components.model.init_from 的配置，非空时 kind 必须为 backbone，path 指向 checkpoint 文件。

        :return: None, 非空初始化时把主干参数加载到现有 self.model.bert；为 None 时直接结束。

        该方法加载 CPU 上的 checkpoint，通过 get_task_state 取得任务状态，只提取 model.bert. 开头的项并去掉该前缀，再 strict=True 加载到主干；保留当前任务头，不恢复优化器或训练进度。
        """
        if init_config is None:
            return
        if init_config['kind'] != 'backbone' or not hasattr(self.model, 'bert'):
            raise ValueError('当前模型族初始化只支持具有 bert 属性的模型及 kind=backbone')
        checkpoint = load_checkpoint(init_config['path'], map_location='cpu')
        state = get_task_state(checkpoint)

        if not isinstance(state, dict) or not state:
            raise ValueError('主干来源缺少非空 task_state 字典')

        prefix = 'model.bert.'
        backbone_state = {}

        for name, value in state.items():
            if name.startswith(prefix):
                backbone_name = name[len(prefix):]
                backbone_state[backbone_name] = value

        if not backbone_state:
            raise ValueError('task_state 中没有 model.bert.* 主干参数')

        self.model.bert.load_state_dict(backbone_state, strict=True)
