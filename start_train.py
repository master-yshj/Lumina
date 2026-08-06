import random
import numpy
import torch

from parse.parser import parse_args, load_config
from tasks.registry import TASK_BUILDER_REGISTRY
from experiments import create_experiments_directories, save_config_snapshot

def main():
    args = parse_args()
    config_path = args.config
    resume = args.resume
    config_dict = load_config(config_path)

    # 创建实验目录
    name = config_dict["name"]
    experiment_paths = create_experiments_directories(name)
    # 配置保存目录
    config_dir = experiment_paths["config_dir"]
    config_snapshot_path = save_config_snapshot(config_dict, config_dir)
    experiment_paths["config_snapshot_path"] = config_snapshot_path

    # resume 实验目录复用

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
        resume=resume,
        experiment_paths=experiment_paths,
    )
    trainer = task_components["trainer"]
    train_dataloader = task_components["train_dataloader"]
    fit_kwargs = task_components["fit_kwargs"]
    train_summary = trainer.fit(train_dataloader, **fit_kwargs)


if __name__ == "__main__":
    main()
