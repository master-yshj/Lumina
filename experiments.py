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
    if not isinstance(config_dir, (str, pathlib.Path)):
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

def resolve_resume_experiment_paths(resume_path):
    """
    根据resume_path组装experiment_pathes
    :param resume_path: 规范后的resume路径
    :return: experiment_paths
    """
    # 获取现有目录，组装已有的experiment_paths
    experiment_paths = {}
    checkpoints_dir = resume_path.parent
    experiment_dir = checkpoints_dir.parent
    if not checkpoints_dir.name == "checkpoints":
        raise ValueError(f"checkpoints_dir: {checkpoints_dir} 必须指向一个叫 checkpoints 的文件夹喵!")
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise ValueError(f"experiment_dir: {experiment_dir} 必须存在且必须指向一个目录喵!")
    experiment_paths["checkpoints_dir"] = checkpoints_dir
    experiment_paths["experiment_dir"] = experiment_dir

    config_dir = experiment_dir / "config"
    logs_dir = experiment_dir / "logs"
    metrics_dir = experiment_dir / "metrics"
    figures_dir = experiment_dir / "figures"
    if not config_dir.exists() or not config_dir.is_dir():
        raise ValueError(f"config_dir: {config_dir} 必须存在且必须为目录喵!")
    if not logs_dir.exists() or not logs_dir.is_dir():
        raise ValueError(f"logs_dir: {logs_dir} 必须存在且必须为目录喵!")
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        raise ValueError(f"metrics_dir: {metrics_dir} 必须存在且必须为目录喵!")
    if not figures_dir.exists() or not figures_dir.is_dir():
        raise ValueError(f"figures_dir: {figures_dir} 必须存在且必须为目录喵!")
    experiment_paths["config_dir"] = config_dir
    experiment_paths["logs_dir"] = logs_dir
    experiment_paths["metrics_dir"] = metrics_dir
    experiment_paths["figures_dir"] = figures_dir

    config_snapshot_path = config_dir / "config.yaml"
    if not config_snapshot_path.exists() or not config_snapshot_path.is_file():
        raise ValueError(f"config_snapshot_path: {config_snapshot_path} 必须存在且必须为文件喵!")
    experiment_paths["config_snapshot_path"] = config_snapshot_path

    return experiment_paths