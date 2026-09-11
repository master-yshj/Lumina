
from parse.parser import parse_args, parse_config
from logger.logger import RunLogger
from device import check_device
from tasks.registry import TASK_BUILDER_REGISTRY
from experiments import create_visualize_directories
from visualize.registry import SUPPORTED_VISUALIZERS

EXPECTED_MODE = "visualize"

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
    visualize_directories = create_visualize_directories(experiment_dir, list(SUPPORTED_VISUALIZERS))
    visualize_dir = visualize_directories["visualize_dir"]
    visualize_log_path = visualize_directories["visualize_log_path"]

    logger = RunLogger(visualize_log_path)
    logger.start({
        "运行模式": "VISUALIZE",
        "实验名称": config_dict["experiment"]["name"],
        "任务名称": config_dict["task"],
        "运行设备": config_dict["runtime"]["device"]["type"],
        "可视化配置文件": config_path,
        "训练配置文件": train_config_path,
        "结果目录": visualize_dir,
        "日志文件": visualize_log_path,
        "Checkpoint": config_dict["run"]["checkpoint"]["path"],
    })

    logger.info(f"可视化即将开始喵!")

    device = config_dict["runtime"]["device"]["type"]
    gpu_nums = config_dict["runtime"]["device"]["count"]
    device_summary = check_device(device, gpu_nums)

    logger.info(f"当前设备信息: {device_summary}")

    task_type = config_dict["task"].strip().casefold()
    TASK_BUILDER_CANONICAL_MAP = TASK_BUILDER_REGISTRY.canonical_map
    task_builder = TASK_BUILDER_CANONICAL_MAP[task_type]
    task_components = task_builder(
        config_dict=config_dict,
        logger=logger,
    )

    task = task_components["task"]
    visualizers = task_components["visualizers"]
    model = task_components["model"].to(device)
    tokenizer = task_components["tokenizer"]
    visualize_dataloader = task_components["visualize_dataloader"]
    visualizer_names = task_components["visualizer_names"]
    for idx, visualizer in enumerate(visualizers):
        save_dir = visualize_dir / visualizer_names[idx]
        visualizer.visualize(
            model=model,
            tokenizer=tokenizer,
            dataloader=visualize_dataloader,
            device=device,
            save_dir=save_dir,
            logger=logger,
        )
    visualize_summary = {
        "task": task,
        "visualizers": visualizers,
        "model": model,
        "tokenizer": tokenizer,
        "dataloader": visualize_dataloader,
        "device": device,
        "visualize_dir": visualize_dir,
        "visualizer_names": visualizer_names,
    }
    logger.finish(visualize_summary)

if __name__ == "__main__":
    main()