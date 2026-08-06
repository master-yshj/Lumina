import pathlib
import time
from copy import deepcopy

import yaml

def create_experiments_directories(experiments_name, experiments_root=None):
    """
    创建实验目录
    :param experiments_name: config.name
    :param experiments_root: 定位到experiments
    :return: directories = {
            "experiment_dir": experiment_dir,
            "config_dir": config_dir,
            "checkpoints_dir": checkpoints_dir,
            "logs_dir": logs_dir,
            "metrics_dir": metrics_dir,
            "figures_dir": figures_dir,
        }
    """
    if not isinstance(experiments_name, str):
        raise ValueError(f"传入的 experiments_name: {experiments_name} 必须为字符串类型的喵! 现在的type: {type(experiments_name)}")
    if not experiments_name.strip():
        raise ValueError(f"传入的 experiments_name 不能为空喵!")
    normalized_name = experiments_name.strip()
    if normalized_name == "." or normalized_name == "..":
        raise ValueError(f"实验名称不能为 '.' 或 '..' 喵!")
    for char in normalized_name:
        if char == "/" or char == "\\":
            raise ValueError(f"实验名称不能包含 '/' 和 '\\' 等非法字符喵!")
    if experiments_root is None:
        experiments_root = pathlib.Path(__file__).resolve().parent / "experiments"
    else:
        if not type(experiments_root) in (str, pathlib.Path):
            raise ValueError(f"experiments_root 在不为None时必须为字符串类型或者pathlib.Path类型的喵! 现在的type: {type(experiments_root)}")
        if isinstance(experiments_root, str) and not experiments_root.strip():
            raise ValueError(f"experiments_root 在为字符串类型时必须不能为空喵!")
    normalized_root = pathlib.Path(experiments_root).resolve()

    current_time = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = normalized_root / normalized_name / current_time
    num = 1
    while experiment_dir.exists():
        experiment_dir = normalized_root / normalized_name / (current_time + f"_{num:02d}")
        num += 1
    experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    child_directories = [
        "config",
        "checkpoints",
        "logs",
        "metrics",
        "figures",
    ]

    directories = {
        "experiment_dir": experiment_dir,
    }

    for child_name in child_directories:
        child_dir = experiment_dir / child_name
        child_dir.mkdir(
            parents=True,
            exist_ok=False,
        )
        directories[f"{child_name}_dir"] = child_dir
    return directories

def transfer_path_to_str(config_dict):
    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                transfer_path_to_str(value)
            if isinstance(value, pathlib.Path):
                config_dict[key] = str(value)
            if isinstance(value, list):
                transfer_path_to_str(value)
    if isinstance(config_dict, list):
        for index, value in enumerate(config_dict):
            if isinstance(value, dict):
                transfer_path_to_str(value)
            if isinstance(value, pathlib.Path):
                config_dict[index] = str(value)
            if isinstance(value, list):
                transfer_path_to_str(value)

def save_config_snapshot(config_dict, config_dir):
    """
    保存配置参数快照
    :param config_dict: 配置参数字典
    :param config_dir: config保存目录
    :return:
    """
    if not isinstance(config_dict, dict):
        raise ValueError(f"config_dict 必须为dict类型的喵! 现在的type: {type(config_dict)}")
    if not config_dict:
        raise ValueError(f"config_dict 不能为空喵!")
    if not type(config_dir) in (str, pathlib.Path):
        raise ValueError(f"config_dir: {config_dir} 必须为字符串类型或者pathlib.Path类型的喵! 现在的type: {type(config_dir)}")
    if isinstance(config_dir, str):
        if not config_dir.strip():
            raise ValueError(f"config_dir 在为字符串类型时不能为空喵!")
    config_dir = pathlib.Path(config_dir)
    if not config_dir.exists():
        raise ValueError(f"config_dir: {config_dir} 指向的路径不存在喵!")
    if not config_dir.is_dir():
        raise ValueError(f"config_dir: {config_dir} 必须指向目录喵!")
    new_dict = deepcopy(config_dict)
    save_path = config_dir / "config.yaml"
    transfer_path_to_str(new_dict)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_dict, f, allow_unicode=True)
    return save_path