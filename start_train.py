import random
import numpy
import torch

from parse.parser import parse_args, load_config, normalize_resume_path
from tasks.registry import TASK_BUILDER_REGISTRY
from experiments import create_experiments_directories, save_config_snapshot, resolve_resume_experiment_paths
from logger.logger import RunLogger

def main():
    args = parse_args()
    config_path = args.config
    resume = args.resume
    config_dict = load_config(config_path)

    # resume 实验目录复用
    resume_path = normalize_resume_path(resume)

    if resume_path is None:
        # 创建实验目录
        name = config_dict["name"]
        experiment_paths = create_experiments_directories(name)
        # 配置保存目录
        config_dir = experiment_paths["config_dir"]
        config_snapshot_path = save_config_snapshot(config_dict, config_dir)
        experiment_paths["config_snapshot_path"] = config_snapshot_path
    else:
        experiment_paths = resolve_resume_experiment_paths(resume_path)

    # 创建日志
    log_path = experiment_paths["logs_dir"] / "train.log"
    logger = RunLogger(log_path)
    logger.start({
        "运行模式": "TRAIN" if resume_path is None else "RESUME",
        "实验名称": config_dict["name"],
        "任务名称": config_dict["task"],
        "随机种子": config_dict["seed"],
        "运行设备": config_dict["cuda"]["device"],
        "配置文件": config_path,
        "实验目录": experiment_paths["experiment_dir"],
        "日志文件": log_path,
        "指标目录": experiment_paths["metrics_dir"],
        "Checkpoint": resume_path,
    })

    logger.info("开始设置随机种子了喵~")

    # 设置种子
    seed = config_dict["seed"]
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # 获取任务类型
    task_type = config_dict["task"].strip().casefold()
    task_builder = TASK_BUILDER_REGISTRY[task_type]
    task_components = task_builder(
        config_dict,
        resume=resume_path,
        experiment_paths=experiment_paths,
        logger=logger,
    )
    trainer = task_components["trainer"]
    train_dataloader = task_components["train_dataloader"]
    fit_kwargs = task_components["fit_kwargs"]
    train_summary = trainer.fit(train_dataloader, **fit_kwargs)

    logger.finish(train_summary)

if __name__ == "__main__":
    main()
