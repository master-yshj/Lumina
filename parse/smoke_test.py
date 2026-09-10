"""Parse 重构 Demo 的 Smoke Test。

PARSE_REFACTOR_DEMO:
    本文件只调用 Parse pipeline，不调用任何 Builder，不创建运行时对象。
    每个可运行用例都读取 cfg/smoke_test 下的独立 YAML，并且每次只向
    标准输出打印一个 Python dict，便于人工检查 YAML 同构结果。
"""

import contextlib
import io
import pathlib
import pprint
import sys


# 允许用户从项目根目录或当前 parse 目录直接执行本文件。
PROJECT_MAIN = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_MAIN) not in sys.path:
    sys.path.insert(0, str(PROJECT_MAIN))

from parse.context import ConfigPath
from parse.errors import ConfigError
from parse.parser import parse_args, parse_config_demo
from parse.schema.merge import merge_schema
from parse.schema.nodes import MappingSchema, ScalarSchema


SMOKE_TEST_CONFIGS = (
    ("mlm_train", "mlm_train.yaml", "train"),
    ("mlm_test", "mlm_test.yaml", "test"),
    ("mlm_inference", "mlm_inference.yaml", "inference"),
    ("qa_train", "qa_train.yaml", "train"),
    ("qa_test", "qa_test.yaml", "test"),
)


def _run_smoke_tests():
    """运行全部可运行 Parse 用例，并返回最后一个规范化配置。"""
    normalized_configs = {}
    smoke_test_directory = PROJECT_MAIN / "cfg" / "smoke_test"
    for test_name, file_name, expected_mode in SMOKE_TEST_CONFIGS:
        config_path = smoke_test_directory / file_name
        # PARSE_REFACTOR_DEMO: 每个用例独立执行 Parse；不复用 Context，
        # 避免不同 task 的选择结果相互污染。
        normalized_config, _, _ = parse_config_demo(
            config_path=config_path,
            expected_mode=expected_mode,
            validate_runtime=False,
        )
        normalized_configs[test_name] = normalized_config
    return normalized_configs


def _run_unit_smoke_tests():
    """验证命令行接口、路径、Schema 合并和异常边界。"""
    # PARSE_REFACTOR_DEMO: 使用 argv 显式注入，验证外部调用方无需修改
    # sys.argv 即可测试 parse_args()。
    args = parse_args(["--config", "config.yaml", "--resume", "checkpoint.pt"])
    assert args.config == "config.yaml"
    assert args.resume == "checkpoint.pt"

    # PARSE_REFACTOR_DEMO: PathSchema 的绝对化结果由上面的配置用例覆盖；
    # 这里检查解析结果不会反向修改输入字典。
    config = {"value": {"nested": 1}}
    schema = MappingSchema(
        fields={"value": MappingSchema(
            fields={"nested": ScalarSchema(python_types=(int,), required=True)},
            allow_unknown=False,
            allow_none=False,
        )},
        allow_unknown=False,
        allow_none=False,
    )
    normalized = schema.normalize(config, ConfigPath.root("config"), None)
    assert normalized == config
    assert normalized is not config

    merged = merge_schema(
        ScalarSchema(python_types=(int,), required=True),
        ScalarSchema(python_types=(int,), required=True),
        ConfigPath.root("config").child("value"),
        "smoke_left",
        "smoke_right",
    )
    assert isinstance(merged, ScalarSchema)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--config", "config.yaml", "--unknown", "value"])
    except SystemExit:
        pass
    else:
        raise AssertionError("parse_args 应拒绝未知命令行参数")

    try:
        parse_config_demo(
            PROJECT_MAIN / "cfg" / "smoke_test" / "mlm_train.yaml",
            expected_mode="test",
            validate_runtime=False,
        )
    except ConfigError:
        pass
    else:
        raise AssertionError("expected_mode 不匹配时应抛出 Parse 领域错误")


def _run_all_smoke_tests():
    """运行配置级和 Parse 基础能力 Smoke Test。"""
    normalized_configs = _run_smoke_tests()
    _run_unit_smoke_tests()
    return normalized_configs


if __name__ == "__main__":
    # PARSE_REFACTOR_DEMO: 输出必须只有一个 Python dict；assert 失败或
    # Parse 异常会直接终止，成功时只输出所有配置的规范化结果。
    pprint.pprint(_run_all_smoke_tests(), sort_dicts=False)
