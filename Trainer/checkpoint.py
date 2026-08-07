import pathlib
import torch
from copy import deepcopy

def build_checkpoint(task, optimizer, scheduler, trainer):
    """
    创建检查点所需状态参数的字典
    :param task: 任务
    :param optimizer: 优化器
    :param scheduler: 调度器
    :param trainer: 训练主框架
    :return: state_dict = {
            "format_version": 1,
            "task_state": task_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "training_progress": training_progress,
        }
    """
    if not trainer.task is task:
        raise ValueError(f"trainer.task: {trainer.task} 与task: {task} 不为同一个对象喵!")
    if not trainer.optimizer is optimizer:
        raise ValueError(f"trainer.optimizer: {trainer.optimizer} 与optimizer: {optimizer} 不为同一个对象喵!")
    if not trainer.scheduler is scheduler:
        raise ValueError(f"trainer.scheduler: {trainer.scheduler} 与scheduler: {scheduler} 不为同一个对象喵!")
    if scheduler is not None:
        if not scheduler.optimizer is optimizer:
            raise ValueError(f"scheduler.optimizer: {scheduler.optimizer} 与optimizer: {optimizer} 不为同一个对象喵!")

    task_state = deepcopy(task.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    scheduler_state = deepcopy(scheduler.state_dict()) if scheduler is not None else None
    training_progress = deepcopy(trainer.get_training_progress())

    state_dict = {
        "format_version": 1,
        "task_state": task_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "training_progress": training_progress,
    }

    return state_dict

def load_checkpoint_state(checkpoint, task, optimizer, scheduler, trainer):
    """
    加载ckpt
    :param checkpoint: build_checkpoint 完整返回的字典内容
    :param task: 任务
    :param optimizer: 优化器
    :param scheduler: 调度器
    :param trainer: 训练主框架
    :return:
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 必须为dict类型的喵! 现在的type: {type(checkpoint)}")
    if not checkpoint:
        raise ValueError(f"checkpoint 不能为空喵!")
    allowed_keys = {"format_version", "task_state", "optimizer_state", "scheduler_state", "training_progress"}
    if "format_version" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 format_version 字段喵!")
    if "task_state" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 task_state 字段喵!")
    if "optimizer_state" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 optimizer_state 字段喵!")
    if "scheduler_state" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 scheduler_state 字段喵!")
    if "training_progress" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 training_progress 字段喵!")
    extra_keys = checkpoint.keys() - allowed_keys
    if extra_keys:
        raise ValueError(f"checkpoint 中多出了其他字段喵: {extra_keys}")
    format_version = checkpoint["format_version"]
    task_state = checkpoint["task_state"]
    optimizer_state = checkpoint["optimizer_state"]
    scheduler_state = checkpoint["scheduler_state"]
    training_progress = checkpoint["training_progress"]
    if not type(format_version) is int or format_version != 1:
        raise ValueError(f"format_version: {format_version} 必须严格为整数1喵!")
    if not isinstance(task_state, dict):
        raise ValueError(f"task_state 必须为dict类型的喵! 现在的type: {type(task_state)}")
    if not task_state:
        raise ValueError(f"task_state 不能为空喵!")
    if not isinstance(optimizer_state, dict):
        raise ValueError(f"optimizer_state 必须为dict类型的喵! 现在的type: {type(optimizer_state)}")
    if not optimizer_state:
        raise ValueError(f"optimizer_state 不能为空喵!")

    if not trainer.task is task:
        raise ValueError(f"trainer.task: {trainer.task} 与task: {task} 不为同一个对象喵!")
    if not trainer.optimizer is optimizer:
        raise ValueError(f"trainer.optimizer: {trainer.optimizer} 与optimizer: {optimizer} 不为同一个对象喵!")
    if not trainer.scheduler is scheduler:
        raise ValueError(f"trainer.scheduler: {trainer.scheduler} 与scheduler: {scheduler} 不为同一个对象喵!")
    if scheduler is not None:
        if not scheduler.optimizer is optimizer:
            raise ValueError(f"scheduler.optimizer: {scheduler.optimizer} 与optimizer: {optimizer} 不为同一个对象喵!")
    if scheduler is None:
        if scheduler_state is not None:
            raise ValueError(f"scheduler 为None时, scheduler_state 也必须为None喵!")
    else:
        if scheduler_state is None:
            raise ValueError(f"scheduler_state 为None时, scheduler 也必须为None喵!")
    if scheduler_state is not None:
        if not isinstance(scheduler_state, dict):
            raise ValueError(f"scheduler_state 不为空时必须为dict类型的喵! 现在的type: {type(scheduler_state)}")

    task.load_state_dict(task_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    trainer.load_training_progress(training_progress)

def save_checkpoint(checkpoint, checkpoint_path):
    """
    保存检查点
    :param checkpoint: build_checkpoint 返回的完整字典
    :param checkpoint_path: checkpoint 检查点保存的.pt的路径
    :return:
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 必须为dict类型的喵! 现在的type: {type(checkpoint)}")
    if not checkpoint:
        raise ValueError(f"checkpoint 不能为空喵!")
    allowed_keys = {"format_version", "task_state", "optimizer_state", "scheduler_state", "training_progress"}
    if "format_version" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 format_version 字段喵!")
    if "task_state" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 task_state 字段喵!")
    if "optimizer_state" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 optimizer_state 字段喵!")
    if "scheduler_state" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 scheduler_state 字段喵!")
    if "training_progress" not in checkpoint:
        raise ValueError(f"checkpoint 中缺少了 training_progress 字段喵!")
    extra_keys = checkpoint.keys() - allowed_keys
    if extra_keys:
        raise ValueError(f"checkpoint 中多出了其他字段喵: {extra_keys}")

    if not type(checkpoint_path) in (str, pathlib.Path):
        raise ValueError(f"checkpoint_path 必须为pathlib.Path类型或字符串类型的喵! 现在的type: {type(checkpoint_path)}")
    if isinstance(checkpoint_path, str):
        if not checkpoint_path.strip():
            raise ValueError(f"checkpoint_path 不能为空喵!")
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.suffix.casefold() == ".pt":
        raise ValueError(f"传入的 checkpoint_path 应为pt模型权重文件保存的路径喵! 现在是: {checkpoint_path}")
    if checkpoint_path.exists():
        if checkpoint_path.is_dir():
            raise ValueError(f"checkpoint_path: {checkpoint_path} 不应该指向一个目录喵!")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, map_location="cpu"):
    """
    加载具体.pt文件
    :param checkpoint_path: 保存的.pt路径
    :param map_location:
    :return: checkpoint: dict 加载的权重字典
    """
    if not type(checkpoint_path) in (str, pathlib.Path):
        raise ValueError(f"checkpoint_path 必须为字符串类型或pathlib.Path类型的喵! 现在的type: {type(checkpoint_path)}")
    if isinstance(checkpoint_path, str):
        if not checkpoint_path.strip():
            raise ValueError(f"checkpoint_path 为字符串类型时不能为空喵!")
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.suffix.casefold() == ".pt":
        raise ValueError(f"传入的 checkpoint_path 应为pt模型权重文件保存的路径喵! 现在是: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise ValueError(f"checkpoint_path: {checkpoint_path} 该路径不存在喵!")
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint_path: {checkpoint_path} 必须指向文件喵!")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 必须为dict类型的喵! 现在的type: {type(checkpoint)}")
    if not checkpoint:
        raise ValueError(f"checkpoint 不能为空喵!")
    return checkpoint