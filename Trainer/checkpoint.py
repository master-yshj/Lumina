import pathlib
import torch
import tempfile
import os
from copy import deepcopy
from math import isfinite
from Trainer.precision import validate_precision_state

def _validate_training_progress(training_dict):
    if not isinstance(training_dict, dict):
        raise TypeError(f"training_dict: {training_dict} 必须为dict类型的喵! 现在的type: {type(training_dict)}")
    expected_keys = {'global_step', 'micro_step', 'completed_validations', 'best_validation_loss', 'validation_rounds_without_improvement', 'update_attempts', 'skipped_updates', 'consecutive_skipped_updates'}
    missing_keys = expected_keys - set(training_dict)
    extra_keys = set(training_dict) - expected_keys
    if missing_keys:
        raise ValueError(f"training_dict 中缺失了键: {missing_keys} 喵!")
    if extra_keys:
        raise ValueError(f"training_dict 中多余了键: {extra_keys} 喵!")
    count_keys = {'global_step', 'micro_step', 'completed_validations', 'validation_rounds_without_improvement', 'update_attempts', 'skipped_updates', 'consecutive_skipped_updates'}
    best = training_dict['best_validation_loss']
    for key in count_keys:
        value = training_dict[key]
        if not type(value) is int or value < 0:
            raise ValueError(f"当前key: {key} 的值value: {value} 不合法喵!")
        if best is not None:
            if not type(best) in [float, int] or not isfinite(best):
                raise ValueError(f"training_dict.best_validation_loss: {best} 必须为有限数值喵!")
    global_step = training_dict['global_step']
    micro_step = training_dict['micro_step']
    completed_validations = training_dict['completed_validations']
    validation_rounds_without_improvement = training_dict['validation_rounds_without_improvement']
    update_attempts = training_dict['update_attempts']
    skipped_updates = training_dict['skipped_updates']
    consecutive_skipped_updates = training_dict['consecutive_skipped_updates']
    if not micro_step>=global_step:
        raise ValueError(f"micro_step: {micro_step} 不能小于 global_step: {global_step} 喵!")
    if not validation_rounds_without_improvement<=completed_validations:
        raise ValueError(f"validation_rounds_without_improvement: {validation_rounds_without_improvement}"
                         f"不能大于"
                         f"completed_validations: {completed_validations} 喵!")
    if completed_validations == 0:
        if best is not None:
            raise ValueError(f"当completed_validations 为0时, best_validation_loss 必须为空喵!")
    else:
        if best is None:
            raise ValueError(f"当completed_validations 不为0时, best_validation_loss 不能为空喵!")
    if not update_attempts==global_step+skipped_updates:
        raise ValueError(f"update_attempts 必须等于 global_step + skipped_updates 喵!\n"
                         f"update_attempts: {update_attempts}\n"
                         f"skipped_updates: {skipped_updates}\n"
                         f"global_step: {global_step}\n")
    if not micro_step>=update_attempts:
        raise ValueError(f"micro_step: {micro_step} 不能小于 update_attempts: {update_attempts} 喵!")
    if not consecutive_skipped_updates<=skipped_updates:
        raise ValueError(f"consecutive_skipped_updates: {consecutive_skipped_updates} 不能大于skipped_updates: {skipped_updates} 喵!")
    return deepcopy(training_dict)

def _validate_checkpoint_structure(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint: {checkpoint} 必须为dict类型的喵! 现在的type: {type(checkpoint)}")
    if not checkpoint:
        raise ValueError(f"checkpoint 不能为空喵!")
    if not "format_version" in checkpoint:
        raise ValueError(f"checkpoint 缺失了 format_version 字段喵!")
    version = checkpoint["format_version"]
    if not type(version) is int or not version in (1, 2):
        raise ValueError(f"format_version: {version} 必须为整型, 且数值必须为1或2喵!")
    expected_keys = {'format_version', 'task_state', 'optimizer_state', 'scheduler_state', 'training_progress'}
    if version == 2:
        expected_keys = expected_keys | {'precision_state', 'runtime_state', 'data_progress'}
    missing_keys = expected_keys - set(checkpoint)
    extra_keys = set(checkpoint) - expected_keys
    if missing_keys:
        raise ValueError(f"checkpoint 中缺失了键: {missing_keys} 喵!")
    if extra_keys:
        raise ValueError(f"checkpoint 中多余了键: {extra_keys} 喵!")
    task_state = checkpoint["task_state"]
    optimizer_state = checkpoint["optimizer_state"]
    scheduler_state = checkpoint["scheduler_state"]
    training_progress = checkpoint["training_progress"]
    if not isinstance(task_state, dict):
        raise TypeError(f"task_state: {task_state} 必须为dict类型的喵! 现在的type: {type(task_state)}")
    if not task_state:
        raise ValueError(f"task_state 不能为空喵!")
    task_state_keys = set(task_state)
    if not all(isinstance(task_state_key, str) for task_state_key in task_state_keys):
        raise ValueError(f"task_state_keys: {task_state_keys} 必须全部为str类型的喵!")
    if not isinstance(optimizer_state, dict):
        raise TypeError(f"optimizer_state: {optimizer_state} 必须为dict类型的喵! 现在的type: {type(optimizer_state)}")
    if not scheduler_state is None and not isinstance(scheduler_state, dict):
        raise TypeError(f"scheduler_state: {scheduler_state} 必须为None或者dict类型的喵! 现在的type: {type(scheduler_state)}")
    if not isinstance(training_progress, dict):
        raise TypeError(f"training_progress: {training_progress} 必须为dict类型的喵! 现在的type: {type(training_progress)}")
    if version == 1:
        progress_copy = deepcopy(training_progress)
        progress_copy['update_attempts'] = progress_copy['global_step']
        progress_copy['skipped_updates']=0
        progress_copy['consecutive_skipped_updates']=0
        _validate_training_progress(progress_copy)
    elif version == 2:
        validate_precision_state(checkpoint['precision_state'])
        _validate_training_progress(training_progress)
        runtime_state = checkpoint['runtime_state']
        if not isinstance(runtime_state, dict):
            raise TypeError(f"runtime_state: {runtime_state} 必须为dict类型的喵! 现在的type: {type(runtime_state)}")
        if not set(runtime_state) == {'strategy', 'world_size', 'loss_reduction'}:
            raise ValueError(f"runtime_state中键: {set(runtime_state)} 必须精确为strategy, world_size, loss_reduction 喵!")
        strategy = runtime_state['strategy']
        world_size = runtime_state['world_size']
        loss_reduction = runtime_state['loss_reduction']
        if not isinstance(strategy, str):
            raise TypeError(f"strategy: {strategy} 必须为str类型的喵! 现在的type: {type(strategy)}")
        if not strategy in ("none", "ddp"):
            raise ValueError(f"strategy: {strategy} 的值只能为none或ddp喵!")
        if not type(world_size) is int or world_size <= 0:
            raise ValueError(f"world_size: {world_size} 必须为正整数喵!")
        if strategy == "ddp":
            if not world_size >= 2:
                raise ValueError(f"strategy 为 ddp 时, world_size 至少需要为2喵!")
        if strategy == "none":
            if not world_size == 1:
                raise ValueError(f"strategy 为 none 时, world_size 必须为1喵!")
        if not loss_reduction == "mean_of_microbatch_means":
            raise ValueError(f"loss_reduction: {loss_reduction} 必须为 mean_of_microbatch_means 喵!")
        data_progress = checkpoint['data_progress']
        if not isinstance(data_progress, dict):
            raise TypeError(f"data_progress: {data_progress} 必须为dict类型的喵! 现在的type: {type(data_progress)}")
        if not set(data_progress) == {"next_epoch", "resume_policy"}:
            raise ValueError(f"data_progress 中键: {set(data_progress)} 必须精确为 next_epoch, resume_policy 喵!")
        next_epoch = data_progress["next_epoch"]
        if not type(next_epoch) is int or next_epoch < 0:
            raise ValueError(f"next_epoch: {next_epoch} 必须为非负整数喵!")
        resume_policy = data_progress["resume_policy"]
        if not resume_policy == "restart_epoch":
            raise ValueError(f"resume_policy: {resume_policy} 目前只能为 restart_epoch 喵!")
    return version

def get_task_state(checkpoint):
    _validate_checkpoint_structure(checkpoint)
    task_state = checkpoint['task_state']
    return task_state

def build_checkpoint(task, optimizer, scheduler, trainer):
    """
    创建检查点所需状态参数的字典
    :param task: 任务
    :param optimizer: 优化器
    :param scheduler: 调度器
    :param trainer: 训练主框架
    :return: state_dict = {
            "format_version": 2,
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
    training_progress=_validate_training_progress(trainer.get_training_progress())
    precision_state=trainer.precision.state_dict()
    context = trainer.runtime_context
    strategy = "ddp" if context.enabled else "none"
    runtime_state = {
        'strategy': strategy,
        'world_size': context.world_size,
        'loss_reduction': 'mean_of_microbatch_means'
    }
    data_progress = {
        'next_epoch': trainer.data_epoch + 1,
        'resume_policy': 'restart_epoch'
    }

    state_dict = {
        "format_version": 2,
        "task_state": task_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "training_progress": training_progress,
        "precision_state": precision_state,
        "data_progress": data_progress,
        "runtime_state": runtime_state,
    }
    _validate_checkpoint_structure(state_dict)

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
    version = _validate_checkpoint_structure(checkpoint)

    if trainer.task is not task:
        raise ValueError("trainer.task 与传入的 task 不是同一个对象喵!")

    if trainer.optimizer is not optimizer:
        raise ValueError("trainer.optimizer 与传入的 optimizer 不是同一个对象喵!")

    if trainer.scheduler is not scheduler:
        raise ValueError("trainer.scheduler 与传入的 scheduler 不是同一个对象喵!")

    if scheduler is not None:
        if scheduler.optimizer is not optimizer:
            raise ValueError("scheduler.optimizer 与传入的 optimizer 不是同一个对象喵!")

    # task_state、optimizer_state 为 dict；scheduler_state 为 dict 或 None。
    task_state = checkpoint["task_state"]
    optimizer_state = checkpoint["optimizer_state"]
    scheduler_state = checkpoint["scheduler_state"]
    context = trainer.runtime_context

    if scheduler is None:
        if scheduler_state is not None:
            raise ValueError(
                "当前没有使用 scheduler，但 checkpoint 中保存了 scheduler_state，"
                "无法完整恢复训练喵!"
            )
    else:
        if scheduler_state is None:
            raise ValueError(
                "当前使用了 scheduler，但 checkpoint 中的 scheduler_state 为 None，"
                "缺少恢复调度器所需的状态喵!"
            )

    if version == 1:
        if not context.enabled and trainer.precision.precision == "fp32":
            progress_copy = deepcopy(checkpoint['training_progress'])
            update_attempts = checkpoint['update_attempts']
            skipped_updates = 0
            consecutive_skipped_updates = 0
            next_epoch = 0
            precision_state = None
    elif version == 2:
        runtime_state = checkpoint['runtime_state']
        precision_state = validate_precision_state(checkpoint['precision_state'])
        strategy = "ddp" if context.enabled else "none"
        if precision_state['precision'] != trainer.precision.precision:
            raise ValueError(
                f"恢复训练时 precision 不一致喵! "
                f"checkpoint 中为 {precision_state['precision']!r}，"
                f"当前为 {trainer.precision.precision!r}"
            )

        if runtime_state['strategy'] != strategy:
            raise ValueError(
                f"恢复训练时 strategy 不一致喵! "
                f"checkpoint 中为 {runtime_state['strategy']!r}，"
                f"当前为 {strategy!r}"
            )

        if runtime_state['world_size'] != context.world_size:
            raise ValueError(
                f"恢复训练时 world_size 不一致喵! "
                f"checkpoint 中为 {runtime_state['world_size']}，"
                f"当前为 {context.world_size}"
            )

        if runtime_state['loss_reduction'] != 'mean_of_microbatch_means':
            raise ValueError(
                f"恢复训练时 loss_reduction 不一致喵! "
                f"checkpoint 中为 {runtime_state['loss_reduction']!r}，"
                f"当前要求为 'mean_of_microbatch_means'"
            )

        progress_copy = deepcopy(checkpoint['training_progress'])

        next_epoch = checkpoint['data_progress']['next_epoch']

        # 当前控制器的 scaler 应为 GradScaler 对象或 None；这里只检查，不创建或加载。
        if precision_state['precision'] == 'fp16':
            if trainer.precision.scaler is None:
                raise ValueError(
                    "checkpoint 使用 fp16，但当前 trainer.precision.scaler 为 None，"
                    "缺少恢复梯度缩放状态所需的 GradScaler 对象喵!"
                )
        else:
            if trainer.precision.scaler is not None:
                raise ValueError(
                    f"checkpoint 使用 {precision_state['precision']!r}，"
                    "按当前框架约定，trainer.precision.scaler 应为 None，"
                    "但当前保存了非 None 对象喵!"
                )
    validated_progress = _validate_training_progress(progress_copy)
    current_task_state = task.state_dict()
    missing_keys = set(task_state) - set(current_task_state)
    extra_keys = set(current_task_state) - set(task_state)
    if missing_keys:
        raise ValueError(f"当前 task_state 中缺失了键: {missing_keys} 喵!")
    if extra_keys:
        raise ValueError(f"当前 task_state 中缺失了键: {extra_keys} 喵!")
    same_keys = set(current_task_state) | set(task_state)
    for key in same_keys:
        current_task_state_value = current_task_state[key]
        task_state_value = task_state[key]
        if isinstance(current_task_state_value, torch.Tensor) and isinstance(task_state_value, torch.Tensor):
            current_task_state_shape = current_task_state_value.shape
            task_state_shape = task_state_value.shape
            if not current_task_state_shape == task_state_shape:
                raise ValueError(f"current_task_state[{key}] 的形状与task_state[{key}] 的形状不一致喵!\n"
                                 f"current_task_state[{key}].shape: {current_task_state_shape}\n"
                                 f"task_state_value[{key}].shape: {task_state_value.shape}\n")
    current_optimizer_state = optimizer.state_dict()

    if 'param_groups' not in optimizer_state:
        raise ValueError("checkpoint 的 optimizer_state 缺少 param_groups 喵!")

    saved_groups = optimizer_state['param_groups']
    if not isinstance(saved_groups, list):
        raise TypeError(
            "checkpoint 的 optimizer_state['param_groups'] 必须为 list 喵! "
            f"当前类型为 {type(saved_groups)}"
        )

    current_groups = current_optimizer_state['param_groups']

    if len(saved_groups) != len(current_groups):
        raise ValueError(
            "Optimizer 的参数组数量不一致喵! "
            f"checkpoint 中有 {len(saved_groups)} 组，"
            f"当前有 {len(current_groups)} 组"
        )

    for group_index, (saved_group, current_group) in enumerate(
            zip(saved_groups, current_groups)
    ):
        if not isinstance(saved_group, dict):
            raise TypeError(
                f"checkpoint 的 param_groups[{group_index}] 必须为 dict 喵!"
            )

        if 'params' not in saved_group:
            raise ValueError(
                f"checkpoint 的 param_groups[{group_index}] 缺少 params 喵!"
            )

        saved_params = saved_group['params']
        if not isinstance(saved_params, list):
            raise TypeError(
                f"checkpoint 的 param_groups[{group_index}]['params'] "
                "必须为 list 喵!"
            )

        current_params = current_group['params']

        if len(saved_params) != len(current_params):
            raise ValueError(
                f"Optimizer 第 {group_index} 组的参数条目数不一致喵! "
                f"checkpoint 中有 {len(saved_params)} 项，"
                f"当前有 {len(current_params)} 项"
            )

    task.load_state_dict(task_state,strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    if version == 2:
        trainer.precision.load_state_dict(precision_state)
    trainer.load_training_progress(validated_progress)
    trainer.data_epoch = next_epoch

def save_checkpoint(checkpoint, checkpoint_path):
    """
    保存检查点
    :param checkpoint: build_checkpoint 返回的完整字典
    :param checkpoint_path: checkpoint 检查点保存的.pt的路径
    :return:
    """
    _validate_checkpoint_structure(checkpoint)

    if isinstance(checkpoint_path ,str):
        if not checkpoint_path.strip():
            raise ValueError(f"checkpoint_path 为str类型时, 不能为空喵!")
        checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.suffix.casefold() == ".pt":
        raise ValueError(f"传入的 checkpoint_path 应为pt模型权重文件保存的路径喵! 现在是: {checkpoint_path}")
    if checkpoint_path.exists():
        if checkpoint_path.is_dir():
            raise ValueError(f"checkpoint_path: {checkpoint_path} 不应该指向一个目录喵!")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=checkpoint_path.name + '.', suffix='.tmp', dir=checkpoint_path.parent,delete=False) as stream:
            temporary_path = pathlib.Path(stream.name)
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, checkpoint_path)
    except Exception as e:
        if temporary_path is not None:
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except Exception as cleanup_error:
                e.add_note(
                    f"清理临时文件 {temporary_path} 失败喵! "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

        raise

def load_checkpoint(checkpoint_path, map_location="cpu"):
    """
    加载具体.pt文件
    :param checkpoint_path: 保存的.pt路径
    :param map_location:
    :return: checkpoint: dict 加载的权重字典
    """
    if not isinstance(checkpoint_path, (str, pathlib.Path)):
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