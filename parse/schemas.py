from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigBase(BaseModel):
    """
    :return: ConfigBase，校验后的配置实例

    该类继承 Pydantic.BaseModel，作为所有配置类的基类；model_config 是类设置，要求拒绝多余字段、校验默认值、使用严格类型并拒绝 inf 和 nan，不是 YAML 字段
    """

    model_config = ConfigDict(extra='forbid', validate_default=True, strict=True, allow_inf_nan=False)


class ExperimentConfig(ConfigBase):
    """
    :param name: str, 必填的实验名称，准备阶段去除首尾空白；不能为空、不能为 . 或 ..，不能包含路径分隔符
    :param seed: int, 随机种子，默认 42，范围为 [0, 4294967295]

    :return: ExperimentConfig，校验后的配置实例

    该类继承 ConfigBase，定义 experiment 配置块，供实验目录命名和随机种子设置使用
    """

    name: str = Field(strict=True, json_schema_extra={"normalize": "strip"})
    seed: int = Field(default=42, strict=True, ge=0, le=4294967295)

    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, value: str) -> str:
        """
        :param cls: type[ExperimentConfig], Pydantic 调用时传入的当前配置类
        :param value: str, 已通过字段类型检查的实验名称

        :return: str, 检查通过后原样返回实验名称

        该函数由 Pydantic 在 name 字段校验后调用，确认名称非空、已去除首尾空白且不是路径，此处不再规范化
        """
        if not value:
            raise ValueError("experiment.name 不能为空喵! 请人工检查实验名称。")
        if value != value.strip():
            raise ValueError("experiment.name 仍含两端空白喵! 请检查准备层是否执行 normalize=strip。")
        if value in (".", ".."):
            raise ValueError("experiment.name 不能为 . 或 .. 喵! 请人工检查实验名称。")
        if "/" in value or "\\" in value:
            raise ValueError("experiment.name 不能包含 / 或反斜杠喵! 请填写名称，不要填写路径。")
        return value


class TrainRunConfig(ConfigBase):
    """
    :param mode: Literal["train"], 必填的运行模式，准备阶段去除首尾空白并 casefold
    :param resume_from: Path | None, 恢复训练的 checkpoint 路径，默认 None 表示不恢复，非空时准备阶段转成绝对 Path

    :return: TrainRunConfig，校验后的配置实例

    该类继承 ConfigBase，定义训练模式专属的 run 配置块，恢复路径的文件和后缀检查由资源校验阶段执行
    """

    mode: Literal["train"] = Field(json_schema_extra={"normalize": "name"})
    resume_from: Path | None = Field(default=None, strict=True, json_schema_extra={"normalize": "path"})


class TaskConfig(ConfigBase):
    """
    :param type: str, 必填的任务名，准备阶段去除首尾空白、casefold 并转换别名

    :return: TaskConfig，校验后的配置实例

    该类继承 ConfigBase，定义 task 配置块；支持的任务名称由 schema_registry 中的任务规则表决定
    """

    type: str = Field(strict=True, json_schema_extra={"normalize": "name"})


class CheckpointInputConfig(ConfigBase):
    """
    :param path: Path, 必填的模型 checkpoint 文件路径，准备阶段转为绝对 Path，资源校验要求存在且为 .pt 文件
    :param train_config: Path, 必填的训练配置 YAML 路径，准备阶段转为绝对 Path，用于继承组件配置；资源类型是 config

    :return: CheckpointInputConfig，校验后的配置实例

    该类继承 ConfigBase，定义 run.checkpoint 配置块，分别记录模型状态文件与训练配置文件
    """

    path: Path = Field(strict=True, json_schema_extra={"normalize": "path", "resource": "checkpoint"})
    train_config: Path = Field(strict=True, json_schema_extra={"normalize": "path", "resource": "config"})


class CheckpointRunConfig(ConfigBase):
    """
    :param mode: Literal["test", "inference", "visualize"], 必填的运行模式，准备阶段去除首尾空白并 casefold
    :param checkpoint: CheckpointInputConfig, 必填的 checkpoint 配置块，输入为包含 path 和 train_config 的字典

    :return: CheckpointRunConfig，校验后的配置实例

    该类继承 ConfigBase，定义测试、推理、可视化三个模式专属的 run 配置块，与 TrainRunConfig 对应不同运行模式
    """

    mode: Literal["test", "inference", "visualize"] = Field(json_schema_extra={"normalize": "name"})
    checkpoint: CheckpointInputConfig

class InitFromConfig(ConfigBase):
    """
    :param kind: Literal["backbone"], 必填的迁移方式，表示从已有 checkpoint 初始化模型主干
    :param path: Path, 必填的初始化 checkpoint 路径，准备阶段转为绝对 Path，资源校验要求存在且为 .pt 文件

    :return: InitFromConfig，校验后的配置实例

    该类继承 ConfigBase，定义 model.init_from 配置块，只描述迁移来源，实际加载由组装层执行
    """

    kind: Literal["backbone"] = Field(json_schema_extra={"normalize": "name"})
    path: Path = Field(strict=True, json_schema_extra={"normalize": "path", "resource": "checkpoint"})


class EmptyParamsConfig(ConfigBase):
    """
    :return: EmptyParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义没有可填写字段的 params，接受空字典并拒绝多余键或 None；能否省略 params 由外层字段的默认值决定
    """


class TokenizerParamsConfig(ConfigBase):
    """
    :param vocab_file_path: Path, 必填的词表文件路径，准备阶段转为绝对 Path，资源校验要求文件存在
    :param do_lower_case: bool, 是否转小写，默认 True
    :param strip_accents: bool | None, 是否去除重音符号，默认 None 表示交给分词器按其规则决定
    :param max_input_chars_per_word: int, 单个词允许的最大字符数，默认 100，必须大于 0

    :return: TokenizerParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 ExampleTokenizer 和 QATokenizer 使用的 params，不在解析阶段创建分词器
    """

    vocab_file_path: Path = Field(strict=True, json_schema_extra={"normalize": "path", "resource": "file"})
    do_lower_case: bool = Field(default=True, strict=True)
    strip_accents: bool | None = Field(default=None, strict=True)
    max_input_chars_per_word: int = Field(default=100, strict=True, gt=0)


class TinyBertParamsConfig(ConfigBase):
    """
    :param model_path: Path | None, 模型族路径，默认 None；非空时只规范化为绝对 Path，metadata 标记表示不检查资源是否存在
    :param hidden_size: int, 隐藏向量维度，默认 128，必须大于 0 且能被 num_heads 整除
    :param max_position_embeddings: int, 位置嵌入表的容量，默认 128，必须大于 0
    :param type_vocab_size: int, token_type 嵌入表的容量，默认 2，必须大于 0
    :param num_heads: int, 注意力头数，默认 4，必须大于 0
    :param intermediate_size: int, 前馈网络中间层维度，默认 512，必须大于 0
    :param num_layers: int, 编码器层数，默认 2，必须大于 0
    :param embedding_dropout: float, 嵌入层丢弃概率，默认 0.1，范围为 [0, 1)
    :param attention_dropout: float, 注意力丢弃概率，默认 0.1，范围为 [0, 1)
    :param hidden_dropout: float, 隐藏状态丢弃概率，默认 0.1，范围为 [0, 1)
    :param layer_norm_eps: float, LayerNorm 的数值稳定项，默认 1.0e-12，必须大于 0

    :return: TinyBertParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 TinyBert 模型的 params；浮点字段在准备阶段统一表示为 float，字段关系在校验阶段检查
    """

    model_path: Path | None = Field(default=None, strict=True, json_schema_extra={"normalize": "path", "resource": "metadata"})
    hidden_size: int = Field(default=128, strict=True, gt=0)
    max_position_embeddings: int = Field(default=128, strict=True, gt=0)
    type_vocab_size: int = Field(default=2, strict=True, gt=0)
    num_heads: int = Field(default=4, strict=True, gt=0)
    intermediate_size: int = Field(default=512, strict=True, gt=0)
    num_layers: int = Field(default=2, strict=True, gt=0)
    embedding_dropout: float = Field(default=0.1, ge=0, lt=1, json_schema_extra={"normalize": "float"})
    attention_dropout: float = Field(default=0.1, ge=0, lt=1, json_schema_extra={"normalize": "float"})
    hidden_dropout: float = Field(default=0.1, ge=0, lt=1, json_schema_extra={"normalize": "float"})
    layer_norm_eps: float = Field(default=1.0e-12, gt=0, json_schema_extra={"normalize": "float"})

    @model_validator(mode='after')
    def validate_head_dimensions(self) -> Self:
        """
        :param self: TinyBertParamsConfig, 已完成各字段校验的当前配置实例

        :return: Self, 维度关系检查通过后返回当前实例

        该函数由 Pydantic 在模型字段校验后调用，检查 hidden_size 能否被 num_heads 整除，保证每个注意力头具有整数维度
        """
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size={self.hidden_size} 必须能被 num_heads={self.num_heads} 整除喵! 请人工检查每个注意力头的维度。")
        return self


class CrossEntropyParamsConfig(ConfigBase):
    """
    :param ignore_index: int, 不参与损失计算的标签值，默认 -100

    :return: CrossEntropyParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义交叉熵损失的 params，供 Builder 构造损失函数使用
    """

    ignore_index: int = Field(default=-100, strict=True)


# BetaValue 是带校验约束的 float 类型别名，范围为 [0, 1)，不接受 inf 或 nan。
BetaValue = Annotated[float, Field(strict=True, ge=0, lt=1, allow_inf_nan=False)]


class AdamWParamsConfig(ConfigBase):
    """
    :param lr: float, 学习率，默认 0.001，必须大于 0
    :param weight_decay: float, 权重衰减系数，默认 0.01，必须大于或等于 0
    :param betas: tuple[float, float], 一阶和二阶矩估计的衰减系数，默认 (0.9, 0.999)，两个值都必须在 [0, 1) 内
    :param eps: float, 数值稳定项，默认 1.0e-8，必须大于 0

    :return: AdamWParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 AdamW 的 params；准备阶段将 betas 列表转为元组并将数值规范为 float，校验阶段检查取值范围
    """

    lr: float = Field(default=0.001, gt=0, json_schema_extra={"normalize": "float"})
    weight_decay: float = Field(default=0.01, ge=0, json_schema_extra={"normalize": "float"})
    betas: tuple[BetaValue, BetaValue] = Field(default=(0.9, 0.999), json_schema_extra={"normalize": "tuple"})
    eps: float = Field(default=1.0e-8, gt=0, json_schema_extra={"normalize": "float"})


class StepLRParamsConfig(ConfigBase):
    """
    :param step_size: int, 每隔多少次调度更新衰减学习率，默认 1000，必须大于 0
    :param gamma: float, 学习率衰减倍率，默认 0.1，必须大于 0
    :param last_epoch: int, PyTorch StepLR 的初始调度计数，默认 -1，必须大于或等于 -1

    :return: StepLRParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义直接使用 StepLR 时的 params，本项目按 step 调度
    """

    step_size: int = Field(default=1000, strict=True, gt=0)
    gamma: float = Field(default=0.1, gt=0, json_schema_extra={"normalize": "float"})
    last_epoch: int = Field(default=-1, strict=True, ge=-1)


class BaseStepLRParamsConfig(ConfigBase):
    """
    :param step_size: int, 基础调度器每隔多少次调度更新衰减学习率，默认 1000，必须大于 0
    :param gamma: float, 学习率衰减倍率，默认 0.1，必须大于 0

    :return: BaseStepLRParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 Warmup 内部 StepLR 的参数，不包含顶层 StepLR 的 last_epoch 字段
    """

    step_size: int = Field(default=1000, strict=True, gt=0)
    gamma: float = Field(default=0.1, gt=0, json_schema_extra={"normalize": "float"})


class WarmupCommonConfig(ConfigBase):
    """
    :param warmup_steps: int, 学习率预热步数，默认 1000，必须大于或等于 0

    :return: WarmupCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义 Warmup 的公共参数，基础调度器名称和参数由 schema_registry 动态添加
    """

    warmup_steps: int = Field(default=1000, strict=True, ge=0)


class TokenAccuracyParamsConfig(ConfigBase):
    """
    :param ignore_index: int, 计算 Token Accuracy 时忽略的标签值，默认 -100

    :return: TokenAccuracyParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 Token Accuracy 指标的 params，使其可与损失函数使用一致的忽略标签
    """

    ignore_index: int = Field(default=-100, strict=True)


class TopKParamsConfig(ConfigBase):
    """
    :param top_k: int, 每个预测位置参与采样的候选 Token 数量，默认 5，必须大于 0

    :return: TopKParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 TopKSamplingStrategy 的 params
    """

    top_k: int = Field(default=5, strict=True, gt=0)


class QASpanParamsConfig(ConfigBase):
    """
    :param max_answer_length: int, 答案跨度允许的最大 Token 数量，默认 30，必须大于 0
    :param n_best_size: int, 起点和终点各保留的候选数量，默认 20，必须大于 0

    :return: QASpanParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 QA 答案跨度解码的 params，供解码策略筛选和组合候选位置
    """

    max_answer_length: int = Field(default=30, strict=True, gt=0)
    n_best_size: int = Field(default=20, strict=True, gt=0)

# AttentionSelector 是类型别名，表示注意力层或头的编号选择规则。
# int 必须为相同的数据类型
# 或者为字符串all
# 或者为一个最小长度为1的list, list里面存储int, 并且要求list里面的数据必须为相同的数据类型, 即int, 作为attention层(头)的编号
AttentionSelector = (
    Annotated[int, Field(strict=True)]
    | Literal["all"]
    | Annotated[list[Annotated[int, Field(strict=True)]], Field(min_length=1)]
)


class AttentionHeatmapParamsConfig(ConfigBase):
    """
    :param layer: int | Literal["all"] | list[int], 要展示的层编号，默认 -1；列表不能为空，字符串会去除首尾空白并 casefold
    :param head: int | Literal["all"] | list[int], 要展示的注意力头编号，默认 0；列表不能为空，字符串会去除首尾空白并 casefold
    :param max_samples: int, 最多展示的样本数量，默认 5，必须大于 0

    :return: AttentionHeatmapParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义注意力热力图的 params，具体层和头的编号是否存在由可视化模块结合模型检查
    """

    layer: AttentionSelector = Field(default=-1, json_schema_extra={"normalize": "name"})
    head: AttentionSelector = Field(default=0, json_schema_extra={"normalize": "name"})
    max_samples: int = Field(default=5, strict=True, gt=0)


class LossLandscapeCommonConfig(ConfigBase):
    """
    :param normalization: Literal["per_parameter", "global"], 扰动方向的归一化方式，默认 per_parameter，另一种为全局归一化
    :param output_format: Literal["png", "html", "both"], 输出文件格式，默认 both
    :param grid_size: int, 每条扰动轴的网格点数量，默认 11，必须大于或等于 2
    :param alpha_min: float, 第一条扰动轴的下界，默认 -1.0，必须小于 alpha_max
    :param alpha_max: float, 第一条扰动轴的上界，默认 1.0
    :param beta_min: float, 第二条扰动轴的下界，默认 -1.0，必须小于 beta_max
    :param beta_max: float, 第二条扰动轴的上界，默认 1.0
    :param batch_count: int, 计算损失使用的批次数量，默认 1，必须大于 0

    :return: LossLandscapeCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义损失曲面可视化的公共 params，losses 字段由 schema_registry 动态添加
    """

    normalization: Literal["per_parameter", "global"] = Field(
        default="per_parameter", json_schema_extra={"normalize": "name"}
    )
    output_format: Literal["png", "html", "both"] = Field(default="both", json_schema_extra={"normalize": "name"})
    grid_size: int = Field(default=11, strict=True, ge=2)
    alpha_min: float = Field(default=-1.0, json_schema_extra={"normalize": "float"})
    alpha_max: float = Field(default=1.0, json_schema_extra={"normalize": "float"})
    beta_min: float = Field(default=-1.0, json_schema_extra={"normalize": "float"})
    beta_max: float = Field(default=1.0, json_schema_extra={"normalize": "float"})
    batch_count: int = Field(default=1, strict=True, gt=0)

    @model_validator(mode='after')
    def validate_ranges(self) -> Self:
        """
        :param self: LossLandscapeCommonConfig, 已完成字段校验的当前配置实例

        :return: Self, 区间检查通过后返回当前实例

        该函数由 Pydantic 在模型字段校验后调用，要求 alpha_min 小于 alpha_max 且 beta_min 小于 beta_max，保证两条扰动轴都有有效区间
        """
        if self.alpha_min >= self.alpha_max:
            raise ValueError(f"alpha_min={self.alpha_min} 必须小于 alpha_max={self.alpha_max} 喵! 请人工检查 alpha 扰动区间。")
        if self.beta_min >= self.beta_max:
            raise ValueError(f"beta_min={self.beta_min} 必须小于 beta_max={self.beta_max} 喵! 请人工检查 beta 扰动区间。")
        return self


class MLMDatasetParamsConfig(ConfigBase):
    """
    :param max_seq_length: int, 包含特殊 Token 的最大样本序列长度，默认 128，必须大于 2

    :return: MLMDatasetParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 MLM 数据集的 params，供文本分块和样本构造使用
    """

    max_seq_length: int = Field(default=128, strict=True, gt=2)


class MLMCollatorParamsConfig(ConfigBase):
    """
    :param mlm_prob: float, MLM 动态掩码使用的抽样概率，默认 0.15，必须大于 0 且小于 1

    :return: MLMCollatorParamsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 MLMCollator 的 params，准备阶段将概率规范为 float
    """

    mlm_prob: float = Field(default=0.15, gt=0, lt=1, json_schema_extra={"normalize": "float"})


class TrainLoaderConfig(ConfigBase):
    """
    :param batch_size: int, 每批样本数量，默认 8，必须大于 0
    :param shuffle: bool, 是否打乱样本顺序，默认 True
    :param num_workers: int, 数据加载工作进程数，默认 0，必须大于或等于 0
    :param pin_memory: bool, 是否使用 DataLoader 的锁页内存功能，默认 False
    :param drop_last: bool, 是否丢弃不足一批的最后一批样本，默认 False

    :return: TrainLoaderConfig，校验后的配置实例

    该类继承 ConfigBase，定义 data.loaders.train 配置块，供 Builder 构造训练 DataLoader
    """

    batch_size: int = Field(default=8, strict=True, gt=0)
    shuffle: bool = Field(default=True, strict=True)
    num_workers: int = Field(default=0, strict=True, ge=0)
    pin_memory: bool = Field(default=False, strict=True)
    drop_last: bool = Field(default=False, strict=True)


class EvalLoaderConfig(ConfigBase):
    """
    :param batch_size: int, 每批样本数量，默认 8，必须大于 0
    :param shuffle: bool, 是否打乱样本顺序，默认 False
    :param num_workers: int, 数据加载工作进程数，默认 0，必须大于或等于 0
    :param pin_memory: bool, 是否使用 DataLoader 的锁页内存功能，默认 False
    :param drop_last: bool, 是否丢弃不足一批的最后一批样本，默认 False

    :return: EvalLoaderConfig，校验后的配置实例

    该类继承 ConfigBase，定义 valid、test 或 visualize 的 DataLoader 配置，当前 inference 不使用该配置块
    """

    batch_size: int = Field(default=8, strict=True, gt=0)
    shuffle: bool = Field(default=False, strict=True)
    num_workers: int = Field(default=0, strict=True, ge=0)
    pin_memory: bool = Field(default=False, strict=True)
    drop_last: bool = Field(default=False, strict=True)


class DatasetSplitsConfig(ConfigBase):
    """
    :param train: Path | None, 训练数据文件路径，默认 None；相对路径以 data.dataset.root 为基准，准备阶段转为绝对 Path
    :param valid: Path | None, 验证数据文件路径，默认 None；相对路径以 data.dataset.root 为基准，准备阶段转为绝对 Path
    :param test: Path | None, 测试数据文件路径，默认 None；相对路径以 data.dataset.root 为基准，准备阶段转为绝对 Path
    :param visualize: Path | None, 可视化数据文件路径，默认 None；相对路径以 data.dataset.root 为基准，准备阶段转为绝对 Path

    :return: DatasetSplitsConfig，校验后的配置实例

    该类继承 ConfigBase，定义 data.dataset.splits；关系校验要求本次启用的划分非空，资源校验只检查本次启用划分的文件
    """

    train: Path | None = Field(
        default=None, strict=True,
        json_schema_extra={"normalize": "path", "path_base": "dataset_root", "resource": "file"}
    )
    valid: Path | None = Field(
        default=None, strict=True,
        json_schema_extra={"normalize": "path", "path_base": "dataset_root", "resource": "file"}
    )
    test: Path | None = Field(
        default=None, strict=True,
        json_schema_extra={"normalize": "path", "path_base": "dataset_root", "resource": "file"}
    )
    visualize: Path | None = Field(
        default=None, strict=True,
        json_schema_extra={"normalize": "path", "path_base": "dataset_root", "resource": "file"}
    )


class LoopConfig(ConfigBase):
    """
    :param total_steps: int, 训练目标的优化器更新步数，对应 global_step，默认 10000，必须大于 0
    :param accumulation_steps: int, 每次优化器更新累积的训练批次数量，默认 1，必须大于 0
    :param log_every_steps: int, 按 global_step 记录训练日志的优化器更新步数间隔，默认 100，必须大于 0

    :return: LoopConfig，校验后的配置实例

    该类继承 ConfigBase，定义 optimization.loop，使训练循环与梯度累积使用统一参数
    """

    total_steps: int = Field(default=10000, strict=True, gt=0)
    accumulation_steps: int = Field(default=1, strict=True, gt=0)
    log_every_steps: int = Field(default=100, strict=True, gt=0)


class EarlyStoppingConfig(ConfigBase):
    """
    :param enabled: bool, 是否启用早停，默认 True
    :param patience: int, 允许验证结果连续未改善的轮数，默认 10，必须大于 0

    :return: EarlyStoppingConfig，校验后的配置实例

    该类继承 ConfigBase，定义 optimization.validation.early_stopping，供训练器按验证结果判断是否停止训练
    """

    enabled: bool = Field(default=True, strict=True)
    patience: int = Field(default=10, strict=True, gt=0)


class CheckpointSaveConfig(ConfigBase):
    """
    :param save_every_rounds: int, 每隔多少轮验证保存 checkpoint，默认 1，必须大于 0

    :return: CheckpointSaveConfig，校验后的配置实例

    该类继承 ConfigBase，定义 optimization.validation.checkpoint 中的周期保存参数
    """

    save_every_rounds: int = Field(default=1, strict=True, gt=0)


class ValidationConfig(ConfigBase):
    """
    :param enabled: bool, 是否启用训练期间的验证，默认 True
    :param every_steps: int, 按 global_step 触发验证的优化器更新步数间隔，默认 500，必须大于 0
    :param early_stopping: EarlyStoppingConfig, 早停配置，省略时从独立空字典开始，再由准备层填充其默认字段
    :param checkpoint: CheckpointSaveConfig, 验证轮次的保存配置，省略时从独立空字典开始，再由准备层填充其默认字段

    :return: ValidationConfig，校验后的配置实例

    该类继承 ConfigBase，定义 optimization.validation，组织验证频率、早停和周期保存参数
    """

    enabled: bool = Field(default=True, strict=True)
    every_steps: int = Field(default=500, strict=True, gt=0)
    early_stopping: EarlyStoppingConfig = Field(default_factory=dict)
    checkpoint: CheckpointSaveConfig = Field(default_factory=dict)


class DeviceConfig(ConfigBase):
    """
    :param type: Literal["cpu", "cuda"], 运行设备类型，默认 cpu，准备阶段去除首尾空白并 casefold
    :param index: int | None, CUDA 设备编号，默认 None；非空时必须大于或等于 0，设备组合另由关系校验检查

    :return: DeviceConfig，校验后的配置实例

    该类继承 ConfigBase，定义 runtime.device，只描述设备选择，不在配置类中查询硬件
    """

    type: Literal["cpu", "cuda"] = Field(default="cpu", json_schema_extra={"normalize": "name"})
    index: int | None = Field(default=None, strict=True, ge=0)


class DistributedConfig(ConfigBase):
    """
    :param strategy: Literal["none", "ddp"], 分布式策略，默认 none
    :param backend: Literal["nccl", "gloo"] | None, 通信后端，默认 None；与策略、设备及运行模式的组合由关系校验检查

    :return: DistributedConfig，校验后的配置实例

    该类继承 ConfigBase，定义 runtime.distributed；准备阶段规范化名称，此处不初始化通信
    """

    strategy: Literal["none", "ddp"] = Field(default="none", json_schema_extra={"normalize": "name"})
    backend: Literal["nccl", "gloo"] | None = Field(default=None, json_schema_extra={"normalize": "name"})


class RuntimeConfig(ConfigBase):
    """
    :param device: DeviceConfig, 设备配置，省略时从独立空字典填充子字段默认值
    :param precision: Literal["fp32", "fp16", "bf16"], 计算精度，默认 fp32，准备阶段去除首尾空白并 casefold
    :param distributed: DistributedConfig, 分布式配置，省略时从独立空字典填充子字段默认值

    :return: RuntimeConfig，校验后的配置实例

    该类继承 ConfigBase，定义 runtime 公共配置块，设备、精度和分布式组合由后续关系校验检查
    """

    device: DeviceConfig = Field(default_factory=dict)
    precision: Literal["fp32", "fp16", "bf16"] = Field(default="fp32", json_schema_extra={"normalize": "name"})
    distributed: DistributedConfig = Field(default_factory=dict)


class MLMInferenceCommonConfig(ConfigBase):
    """
    :param max_seq_length: int, MLM 推理输入使用的最大序列长度，默认 128，必须大于 0

    :return: MLMInferenceCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义 MLM inference 的公共字段，decoding 字段由 schema_registry 动态添加
    """

    max_seq_length: int = Field(default=128, strict=True, gt=0)


class QAInferenceCommonConfig(ConfigBase):
    """
    :param doc_stride: int, QA 推理相邻上下文窗口重叠的 Token 数，必填且必须大于或等于 0，实际窗口容量限制由分词器检查

    :return: QAInferenceCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义 QA inference 的公共字段，decoding 字段由 schema_registry 动态添加
    """

    doc_stride: int = Field(strict=True, ge=0)


class LossCommonConfig(ConfigBase):
    """
    :param name: str, 必填的损失项名称，准备阶段去除首尾空白；不能为空或包含点号
    :param weight: float, 该损失项的加权系数，默认 1.0，必须大于或等于 0

    :return: LossCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义损失项的公共字段，type 和 params 由 schema_registry 动态添加
    """

    name: str = Field(strict=True, json_schema_extra={"normalize": "strip"})
    weight: float = Field(default=1.0, ge=0, json_schema_extra={"normalize": "float"})

    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, value: str) -> str:
        """
        :param cls: type[LossCommonConfig], Pydantic 调用时传入的当前配置类
        :param value: str, 已通过字段类型检查的损失项名称

        :return: str, 检查通过后原样返回名称

        该方法由 Pydantic 在 name 字段校验后调用，检查非空、没有首尾空白且不含点号，满足后续用作 ModuleDict 键的基本要求
        """
        if not value:
            raise ValueError("loss.name 不能为空喵! 请人工检查损失项名称。")
        if value != value.strip():
            raise ValueError("loss.name 仍含两端空白喵! 请检查准备层是否执行 normalize=strip。")
        if "." in value:
            raise ValueError("loss.name 不能包含点号喵! 请人工检查名称，避免与 ModuleDict 的键规则冲突。")
        return value


class ModelCommonConfig(ConfigBase):
    """
    :param init_from: InitFromConfig | None, 主干权重初始化配置，默认 None 表示不从已有 checkpoint 初始化

    :return: ModelCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义 model 的公共字段，type 和 params 由 schema_registry 动态添加
    """

    init_from: InitFromConfig | None = None


class SchedulerCommonConfig(ConfigBase):
    """
    :param interval: Literal["step"], 调度器的更新时机，默认 step，当前只支持按训练更新步调用

    :return: SchedulerCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义 scheduler 的公共字段，type 和 params 由 schema_registry 动态添加
    """

    interval: Literal["step"] = Field(default="step", json_schema_extra={"normalize": "name"})


class RootCommonConfig(ConfigBase):
    """
    :param version: int, 必填的配置格式版本，当前必须等于 2
    :param experiment: ExperimentConfig, 必填的实验配置块，输入为普通字典
    :param task: TaskConfig, 必填的任务配置块，输入为普通字典
    :param runtime: RuntimeConfig, 运行配置，省略时从独立空字典填充子字段默认值

    :return: RootCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义各任务和模式共有的根字段，run、components 等字段由 schema_registry 按任务及模式动态添加
    """

    version: int = Field(strict=True)
    experiment: ExperimentConfig
    task: TaskConfig
    runtime: RuntimeConfig = Field(default_factory=dict)

    @field_validator('version', mode='after')
    @classmethod
    def validate_version(cls, value: int) -> int:
        """
        :param cls: type[RootCommonConfig], Pydantic 调用时传入的当前配置类
        :param value: int, 已通过严格整数类型检查的配置版本号

        :return: int, 检查通过后返回版本号 2

        该类由 Pydantic 在 version 字段校验后调用，拒绝当前解析器不支持的配置格式版本
        """
        if value != 2:
            raise ValueError(f"version={value} 不受支持喵! 当前配置版本必须为 2，请人工检查或迁移旧 YAML")
        return value


class TrainEvaluationCommonConfig(ConfigBase):
    """
    :param test_after_train: bool, 训练完成后是否执行测试集评估，默认 False

    :return: TrainEvaluationCommonConfig，校验后的配置实例

    该类继承 ConfigBase，定义训练模式 evaluation 的公共字段，metrics 字段由 schema_registry 动态添加
    """

    test_after_train: bool = Field(default=False, strict=True)


class ComponentCommonConfig(ConfigBase):
    """
    :return: ComponentCommonConfig，校验后的配置实例

    该类继承 ConfigBase，作为普通组件配置的空基类，供 schema_registry 添加 type 和 params 字段，并继承严格字段校验规则
    """
