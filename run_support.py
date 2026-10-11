
import argparse
import json
from pathlib import Path
import random
import os

import numpy as np
import torch

from device import check_device
from experiments import (create_experiments_directories, resolve_resume_experiment_paths,
                         create_test_directories, create_inference_directories,
                         create_visualize_directories, save_config_snapshot)
from Trainer.checkpoint import load_checkpoint, get_task_state
from tasks.builder.support import visualizer_output_name


def parse_entry_args(expected_mode, argv=None):
    parser = argparse.ArgumentParser(description=f'Lumina {expected_mode} 入口')
    parser.add_argument('--config', required=True, help='version 2 YAML 路径')
    parser.add_argument('--check-config', action='store_true', help='只解析与校验，不创建运行目录')
    if expected_mode == 'train':
        parser.add_argument('--resume', default=argparse.SUPPRESS, help='覆盖 run.resume_from')
    parser.add_argument('overrides', nargs='*', metavar='key=value', help='OmegaConf dotlist 覆盖')
    return parser.parse_args(argv)


def make_cli_overrides(args):
    overrides = list(args.overrides)
    if hasattr(args, 'resume'):
        overrides.append('run.resume_from=' + json.dumps(args.resume, ensure_ascii=False))
    return overrides


def prepare_runtime(config):
    runtime = config['runtime']
    if runtime['distributed']['strategy'] != 'none' or runtime['precision'] != 'fp32':
        raise NotImplementedError('独立测试/推理/可视化首期只支持单进程FP32')
    env_keys = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE')

    missing_keys = [
        key for key in env_keys
        if key not in os.environ
    ]

    if missing_keys and len(missing_keys) != len(env_keys):
        present_keys = [
            key for key in env_keys
            if key in os.environ
        ]
        raise ValueError(
            "独立测试/推理/可视化的进程环境变量不能部分存在喵! "
            f"已存在: {present_keys}; 缺失: {missing_keys}"
        )

    if not missing_keys:
        env_values = {}

        for key in env_keys:
            raw_value = os.environ[key]
            try:
                env_values[key] = int(raw_value, 10)
            except ValueError as error:
                raise ValueError(
                    f"环境变量 {key} 必须是十进制整数喵! "
                    f"当前值: {raw_value!r}"
                ) from error

        process_state = (
            env_values['RANK'],
            env_values['LOCAL_RANK'],
            env_values['WORLD_SIZE'],
            env_values['LOCAL_WORLD_SIZE'],
        )

        if process_state != (0, 0, 1, 1):
            raise ValueError(
                "独立测试/推理/可视化首期只支持单进程，"
                "要求 RANK=0、LOCAL_RANK=0、WORLD_SIZE=1、LOCAL_WORLD_SIZE=1 喵! "
                f"当前值: {env_values}"
            )

    spec = runtime['device']
    backend = spec['type']
    index = spec['index']
    if backend == 'cpu':
        if index is not None:
            raise ValueError('CPU 的 runtime.device.index 必须为 null')
        requested_device = 'cpu'
    elif backend == 'cuda':
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError('CUDA 的 runtime.device.index 必须为非负整数或 null')
        requested_device = 'cuda' if index is None else f'cuda:{index}'
    else:
        raise NotImplementedError(f'当前执行层只支持 CPU 与单卡 CUDA，收到：{backend}')
    summary = check_device(requested_device, gpu_nums=1)
    device = torch.device(summary['backend'], summary['device_index'])
    return device, summary


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_run_outputs(config):
    mode = config['run']['mode']
    metric_files = {}
    snapshot_path = checkpoints_dir = visualize_dir = figures_dir = None
    if mode == 'train':
        resume = config['run']['resume_from']
        if resume is None:
            paths = create_experiments_directories(config['experiment']['name'])
            snapshot_path = save_config_snapshot(config, paths['config_dir'])
        else:
            paths = resolve_resume_experiment_paths(resume)
            snapshot_path = paths['config_snapshot_path']
        experiment_dir = run_dir = paths['experiment_dir']
        log_path = paths['logs_dir'] / 'train.log'
        checkpoints_dir, figures_dir = paths['checkpoints_dir'], paths['figures_dir']
        splits = ['train']
        if config['optimization']['validation']['enabled']:
            splits.append('valid')
        if config['evaluation']['test_after_train']:
            splits.append('test')
        metric_files = {s: paths['metrics_dir'] / f'{s}_metrics.csv' for s in splits}
    else:
        train_config = config['run']['checkpoint']['train_config']
        if not isinstance(train_config, Path) or not train_config.is_file():
            raise ValueError('run.checkpoint.train_config 必须指向实际训练快照文件')
        experiment_dir = train_config.parent.parent
        if mode == 'test':
            paths = create_test_directories(experiment_dir)
            run_dir, log_path = paths['test_dir'], paths['test_logs_path']
            metric_files = {'test': paths['test_metrics_path']}
        elif mode == 'inference':
            paths = create_inference_directories(experiment_dir)
            run_dir, log_path = paths['inference_dir'], paths['inference_logs_path']
        elif mode == 'visualize':
            names = [visualizer_output_name(i, s['type']) for i, s in enumerate(config['visualize']['methods'], 1)]
            paths = create_visualize_directories(experiment_dir, names)
            visualize_dir = run_dir = paths['visualize_dir']
            log_path = paths['visualize_log_path']
        else:
            raise ValueError(f'未知运行模式：{mode}')
    return dict(experiment_dir=experiment_dir, run_dir=run_dir, log_path=log_path,
                snapshot_path=snapshot_path, visualize_dir=visualize_dir, figures_dir=figures_dir,
                output_paths=dict(metric_files=metric_files, checkpoints_dir=checkpoints_dir))


def load_visualization_task_weights(task, checkpoint_path):
    path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(path, map_location='cpu')
    state=get_task_state(checkpoint)
    task.load_state_dict(state, strict=True)
    return path

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    return None