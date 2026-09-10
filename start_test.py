
from parse.parser import parse_args, parse_config
from logger.logger import RunLogger
from device import check_device
from tasks.registry import TASK_BUILDER_REGISTRY
from Tester.test_runner import run_test
from experiments import create_test_directories

EXPECTED_MODE = "test"

def main():
    args = parse_args()
    config_path = args.config
    config_dict, parse_context, compiled_schema = parse_config(
        config_path=config_path,
        expected_mode=EXPECTED_MODE,
        validate_runtime=True,
    )

    train_config_path = config_dict["run"]["checkpoint"]["train_config"]
    checkpoint_path = config_dict["run"]["checkpoint"]["path"]
    train_config_dir = train_config_path.parent
    experiment_dir = train_config_dir.parent
    test_directories = create_test_directories(experiment_dir)
    test_config_path = test_directories["test_config_path"]
    test_logs_path = test_directories["test_logs_path"]
    test_metrics_path = test_directories["test_metrics_path"]
    test_dir = test_directories["test_dir"]

    logger = RunLogger(test_logs_path)
    logger.start({
        "运行模式": "TEST",
        "实验名称": config_dict["experiment"]["name"],
        "任务名称": config_dict["task"],
        "运行设备": config_dict["runtime"]["device"]["type"],
        "测试配置文件": config_path,
        "训练配置文件": train_config_path,
        "结果目录": test_dir,
        "日志文件": test_logs_path,
        "指标目录": test_metrics_path,
        "Checkpoint": config_dict["run"]["checkpoint"]["path"],
    })

    logger.info(f"测试即将开始喵!")

    device = config_dict["runtime"]["device"]["type"]
    gpu_nums = config_dict["runtime"]["device"]["count"]
    device_summary = check_device(device, gpu_nums)

    logger.info(f"当前设备信息: {device_summary}")

    # 获取任务类型
    task_type = config_dict["task"].strip().casefold()
    TASK_BUILDER_CANONICAL_MAP = TASK_BUILDER_REGISTRY.canonical_map
    task_builder = TASK_BUILDER_CANONICAL_MAP[task_type]
    task_components = task_builder(
        config_dict=config_dict,
        test_metrics_path=test_metrics_path,
        logger=logger,
    )
    task = task_components["task"]
    test_dataloader = task_components["test_dataloader"]
    test_metric_collection = task_components["test_metric_collection"]
    test_metrics_writer = task_components["test_metrics_writer"]

    test_summary = run_test(
        task=task,
        test_dataloader=test_dataloader,
        checkpoint_path=checkpoint_path,
        device=device,
        metric_collection=test_metric_collection,
        metrics_writer=test_metrics_writer,
        logger=logger,
    )

    logger.finish(test_summary)

if __name__ == "__main__":
    main()