"""Parse 重构 Demo 的正式流水线入口。

该模块按最终职责顺序连接 YAML 加载、Bootstrap、Schema Builder、Compile、
Normalize 和 Validate。它不会调用任务 Builder，也不会构造训练对象。
"""

import pathlib

from argparse import ArgumentParser, Namespace
from parse.bootstrap import bootstrap_config
from parse.context import ConfigPath
from parse.path import resolve_paths
from parse.errors import YamlLoadError
from parse.schema.builder import build_schema
from parse.schema.compile import compile_schema
from parse.validation.pipeline import validate_config


def _load_registries():
    """加载真实运行 Registry，避免仅导入 Parse 模块就加载运行依赖。"""
    # PARSE_REFACTOR_DEMO: Registry 延迟装配只改变导入时机，不改变 Registry
    # 的所有权；真正执行 Parse 时仍使用项目现有的运行 Registry。
    from Inference.decoding.registry import DECODING_REGISTRY
    from metrics.registry import METRIC_REGISTRY
    from models.Transformer.losses.registry import LOSS_REGISTRY
    from models.Transformer.modeling.registry import MODEL_REGISTRY
    from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
    from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
    from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
    from tasks.registry import TASK_BUILDER_REGISTRY
    # AI-CHANGE: 可视化方法运行 Registry 接入(AI 编写, 待人工审查)。
    # 只装配方法级 Registry,不涉及任务 Builder；模型与数据集契约另行设计。
    from visualize.registry import VISUALIZER_REGISTRY

    return {
        "task": TASK_BUILDER_REGISTRY,
        "tokenizer": TOKENIZER_REGISTRY,
        "model": MODEL_REGISTRY,
        "loss": LOSS_REGISTRY,
        "optimizer": OPTIMIZER_REGISTRY,
        "scheduler": SCHEDULER_REGISTRY,
        "metric": METRIC_REGISTRY,
        "decoding": DECODING_REGISTRY,
        "visualizer": VISUALIZER_REGISTRY,
    }


def _load_spec_registries():
    """加载各动态组件的 ComponentSpec Registry。"""
    # PARSE_REFACTOR_DEMO: Spec Registry 与运行 Registry 分离，Dataset、
    # Collator 和 Runtime 不在这里注册。
    from parse.specs.registry import (
        DECODING_SPEC_REGISTRY,
        LOSS_SPEC_REGISTRY,
        METRICS_SPEC_REGISTRY,
        MODEL_SPEC_REGISTRY,
        OPTIMIZER_SPEC_REGISTRY,
        SCHEDULER_SPEC_REGISTRY,
        TASK_SPEC_REGISTRY,
        TOKENIZER_SPEC_REGISTRY,
        VISUALIZER_SPEC_REGISTRY,
    )

    return {
        "task": TASK_SPEC_REGISTRY,
        "tokenizer": TOKENIZER_SPEC_REGISTRY,
        "model": MODEL_SPEC_REGISTRY,
        "loss": LOSS_SPEC_REGISTRY,
        "optimizer": OPTIMIZER_SPEC_REGISTRY,
        "scheduler": SCHEDULER_SPEC_REGISTRY,
        "metric": METRICS_SPEC_REGISTRY,
        "decoding": DECODING_SPEC_REGISTRY,
        # AI-CHANGE: 可视化方法 Spec Registry 接入(AI 编写, 待人工审查)。
        # 与运行侧 visualizer 类别对应；任务级 visualize Spec 待补充。
        "visualizer": VISUALIZER_SPEC_REGISTRY,
    }


def parse_args(argv=None) -> Namespace:
    """读取 Parse 外部入口需要的命令行参数。

    PARSE_REFACTOR_DEMO: 该接口只解析参数，不读取 YAML、不执行 Bootstrap、
    不执行 normalize，也不调用任何 Builder；argv 参数用于外部测试注入。
    """
    parser = ArgumentParser(
        description="使用命令行运行 Parse 配置，例如：python start_train.py --config cfg/example.yaml"
    )
    parser.add_argument(
        "--config",
        help="要解析的 YAML 配置路径",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resume",
        help="训练恢复检查点路径；由外部 Builder 决定是否使用",
        type=str,
        default=None,
    )
    return parser.parse_args(argv)


def load_yaml(config_path):
    """安全读取 YAML，并把依赖、文件和语法错误转换为 Parse 领域错误。

    PARSE_REFACTOR_DEMO:
        延迟导入 PyYAML，使 Schema、Registry 和 Bootstrap 可以在未安装 YAML
        依赖的环境中独立导入和静态审查；只有加载 YAML 时才要求该依赖。
    """
    config_path = pathlib.Path(config_path).absolute()
    try:
        import yaml
    except ImportError as error:
        raise YamlLoadError("加载YAML配置需要安装PyYAML喵!") from error
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as error:
        raise YamlLoadError(f"配置文件: {config_path} 加载失败喵!") from error
    return raw_config, config_path


def parse_config(config_path, expected_mode=None, validate_runtime=False):
    """解析外部入口传入的配置并返回规范化结果。

    PARSE_REFACTOR_DEMO: 这是训练、测试和推理入口统一调用的 Parse 接口；
    它只负责配置解析，不负责选择或调用任何 Builder。默认关闭实际设备探测，
    使配置结构可以在没有 CUDA 的开发环境中验证。
    """
    raw_config, config_path = load_yaml(config_path)
    runtime_registries = _load_registries()
    spec_registries = _load_spec_registries()
    context = bootstrap_config(
        raw_config=raw_config,
        config_path=config_path,
        task_registry=runtime_registries["task"],
        task_spec_registry=spec_registries["task"],
        component_registries=runtime_registries,
        component_spec_registries=spec_registries,
        expected_mode=expected_mode,
        validate_runtime=validate_runtime,
    )
    root_schema = build_schema(context, runtime_registries)
    compiled_schema = compile_schema(root_schema)
    normalized_config = compiled_schema.root.normalize(
        value=raw_config,
        path=ConfigPath.root("config"),
        context=context,
    )
    resolved_config = resolve_paths(normalized_config, compiled_schema, context)
    validated_config = validate_config(
        resolved_config,
        context,
        validator_refs=compiled_schema.validator_refs,
    )
    return validated_config, context, compiled_schema


# PARSE_REFACTOR_DEMO: 保留旧名称作为 Smoke Test 和已有外部调用方的兼容别名；
# 新的训练、测试、推理入口统一使用 parse_config()。
def parse_config_demo(config_path, expected_mode=None, validate_runtime=False):
    return parse_config(
        config_path=config_path,
        expected_mode=expected_mode,
        validate_runtime=validate_runtime,
    )

def normalize_resume_path(resume):
    """
    将控制台传入的resume参数规范化
    :param resume: resume参数，模型开始训练的ckpt保存的pt文件
    :return:
    """
    if resume is None:
        return None
    if not type(resume) in (str, pathlib.Path):
        raise ValueError(f"resume 必须为字符串类型或 pathlib.Path 类型的喵! 现在的type: {type(resume)}")
    if isinstance(resume, str):
        if not resume.strip():
            raise ValueError(f"resume 为字符串类型时不能为空喵!")
    resume_path = pathlib.Path(resume).resolve()
    if not resume_path.exists():
        raise ValueError(F"resume_path: {resume_path} 指向的路径不存在喵!")
    if not resume_path.is_file():
        raise ValueError(f"resume_path: {resume_path} 必须指向文件喵!")
    if resume_path.suffix.casefold() == ".pt":
        return resume_path
    else:
        raise ValueError(f"resume_path: {resume_path} 必须指向pt类型文件喵!")