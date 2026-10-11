import pathlib

from Trainer.checkpoint import load_checkpoint, get_task_state
from Inference.inferencer import Inferencer

def _load_task_checkpoint(task, checkpoint_path):
    normalized_checkpoint_path = checkpoint_path.resolve()
    checkpoint = load_checkpoint(normalized_checkpoint_path, map_location="cpu")
    state = get_task_state(checkpoint)
    task.load_state_dict(state, strict=True)

    return checkpoint

def _log_info(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info(message)

def run_inference(task, task_inference, checkpoint_path, device, logger=None):
    preferred_path = pathlib.Path(checkpoint_path)
    fallback_path = preferred_path.parent / "last.pt"
    checkpoint_errors = {}
    if preferred_path == fallback_path:
        candidate_paths = [
            preferred_path,
        ]
    else:
        candidate_paths = [
            preferred_path,
            fallback_path,
        ]
    checkpoint = None
    used_checkpoint_path = None
    for candidate_path in candidate_paths:
        try:
            _log_info(logger, f"正在尝试加载checkpoint: {candidate_path} ...")
            loaded_checkpoint = _load_task_checkpoint(task, candidate_path)
        except Exception as e:
            checkpoint_errors[candidate_path] = str(e)
            _log_info(logger,
                      f"加载失败喵! checkpoint: {candidate_path} | 原始错误信息: {checkpoint_errors[candidate_path]}")
        else:
            used_checkpoint_path = candidate_path
            checkpoint = loaded_checkpoint
            _log_info(logger, f"加载成功喵! checkpoint: {used_checkpoint_path}")
            break
    if used_checkpoint_path is None:
        _log_info(logger, f"所有的checkpoint加载均失败喵! checkpoint_errors: {checkpoint_errors}")
        raise ValueError(f"所有的checkpoint加载均失败喵! checkpoint_errors: {checkpoint_errors}")

    inferencer = Inferencer(
        task=task,
        task_inference=task_inference,
        device=device,
        logger=logger,
    )

    summary_dict = {
        "inferencer": inferencer,
        "checkpoint_path": used_checkpoint_path,
    }

    return summary_dict