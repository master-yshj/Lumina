
import pathlib
from parse.parser import parse_config
from runtime.distributed import initialize_training_runtime
from run_support import (
    parse_entry_args,
    make_cli_overrides,
    prepare_run_outputs,
    seed_process,
)
from logger.logger import RunLogger
from tasks.registry import create_builder
import csv
from metrics.output.plotter import plot_csv_metrics
from Tester.test_runner import run_test

EXPECTED_MODE = 'train'


def parse_args(argv=None):
    return parse_entry_args(EXPECTED_MODE, argv)


def main(argv=None):
    args = parse_args(argv)
    config = parse_config(
        args.config,
        cli_overrides=make_cli_overrides(args),
        expected_mode=EXPECTED_MODE,
    )
    if args.check_config:
        print(
            f"配置解析与校验通过：task={config['task']['type']}，"
            f"mode={config['run']['mode']}，未启动运行。"
        )
        return config

    context, device_summary = initialize_training_runtime(config)
    device = context.device
    local_logger = None
    primary_error = None
    summary = None

    try:
        def local_prepare():
            return prepare_run_outputs(config)

        paths = context.run_main('准备实验路径', local_prepare)

        def check_paths():
            if not isinstance(paths, dict):
                raise TypeError(
                    f"prepare_run_outputs 必须返回 dict 喵! 当前类型: {type(paths)}"
                )
            required_keys = {
                'experiment_dir', 'run_dir', 'log_path', 'output_paths',
            }
            missing_keys = required_keys - set(paths)
            if missing_keys:
                raise ValueError(f"paths 缺少必需字段喵! {sorted(missing_keys)}")

            for key in ('experiment_dir', 'run_dir'):
                directory = paths[key]
                if not isinstance(directory, pathlib.Path):
                    raise TypeError(f"paths.{key} 必须为 Path 喵! 当前类型: {type(directory)}")
                if not directory.is_absolute():
                    raise ValueError(f"paths.{key} 必须为绝对路径喵! {directory}")
                if not directory.is_dir():
                    raise ValueError(
                        f"rank {context.rank} 无法访问 paths.{key} 对应的目录喵! {directory}"
                    )

            log_path = paths['log_path']
            if not isinstance(log_path, pathlib.Path):
                raise TypeError(f"paths.log_path 必须为 Path 喵! 当前类型: {type(log_path)}")
            if not log_path.is_absolute():
                raise ValueError(f"paths.log_path 必须为绝对路径喵! {log_path}")
            if not log_path.parent.is_dir():
                raise ValueError(
                    f"rank {context.rank} 无法访问日志父目录喵! {log_path.parent}"
                )
            if log_path.exists() and not log_path.is_file():
                raise ValueError(f"paths.log_path 已存在，但不是文件喵! {log_path}")

            output_paths = paths['output_paths']
            if not isinstance(output_paths, dict):
                raise TypeError(
                    f"paths.output_paths 必须为 dict 喵! 当前类型: {type(output_paths)}"
                )
            missing_output_keys = {'metric_files', 'checkpoints_dir'} - set(output_paths)
            if missing_output_keys:
                raise ValueError(
                    f"paths.output_paths 缺少必需字段喵! {sorted(missing_output_keys)}"
                )

            checkpoints_dir = output_paths['checkpoints_dir']
            if not isinstance(checkpoints_dir, pathlib.Path):
                raise TypeError(
                    "paths.output_paths.checkpoints_dir 必须为 Path 喵! "
                    f"当前类型: {type(checkpoints_dir)}"
                )
            if not checkpoints_dir.is_absolute():
                raise ValueError(f"checkpoints_dir 必须为绝对路径喵! {checkpoints_dir}")
            if not checkpoints_dir.is_dir():
                raise ValueError(
                    f"rank {context.rank} 无法访问检查点目录喵! {checkpoints_dir}"
                )

            metric_files = output_paths['metric_files']
            if not isinstance(metric_files, dict):
                raise TypeError(
                    f"metric_files 必须为 dict 喵! 当前类型: {type(metric_files)}"
                )
            for split, csv_path in metric_files.items():
                field = f"paths.output_paths.metric_files.{split}"
                if not isinstance(csv_path, pathlib.Path):
                    raise TypeError(f"{field} 必须为 Path 喵! 当前类型: {type(csv_path)}")
                if not csv_path.is_absolute():
                    raise ValueError(f"{field} 必须为绝对路径喵! {csv_path}")
                if csv_path.suffix.casefold() != '.csv':
                    raise ValueError(f"{field} 必须以 .csv 结尾喵! {csv_path}")
                if not csv_path.parent.is_dir():
                    raise ValueError(
                        f"rank {context.rank} 无法访问 {field} 的父目录喵! {csv_path.parent}"
                    )
                if csv_path.exists() and not csv_path.is_file():
                    raise ValueError(f"{field} 已存在，但不是文件喵! {csv_path}")
            return None

        context.run_local('检查共享输出目录', check_paths)

        def create_local_logger():
            nonlocal local_logger
            if not context.is_main:
                return None

            local_logger = RunLogger(paths['log_path'])
            start_info = {
                '任务': config['task']['type'],
                '运行模式': EXPECTED_MODE,
                '配置路径': args.config,
                '设备': str(device),
                '实验目录': paths['experiment_dir'],
            }
            local_logger.start(start_info)
            return local_logger

        local_logger = context.run_local('创建运行日志', create_local_logger)

        def seed_model():
            seed_process(config['experiment']['seed'])
            return None

        context.run_local('设置模型种子', seed_model)

        def make_builder():
            return create_builder(
                config,
                device=device,
                device_summary=device_summary,
                output_paths=paths['output_paths'],
                logger=local_logger,
                runtime_context=context,
            )

        builder = context.run_local('创建 Builder', make_builder)

        result = builder.build()

        def seed_training():
            seed_process(config['experiment']['seed'] + context.rank)
            return None

        context.run_local('设置本进程训练种子', seed_training)

        trainer = result['trainer']
        fit_kwargs = result['fit_kwargs']
        train_summary = trainer.fit(**fit_kwargs)

        def postprocess():
            metric_files = paths['output_paths']['metric_files']
            for split in ('train', 'valid'):
                if split not in metric_files:
                    continue
                csv_path = metric_files[split]
                with csv_path.open(encoding='utf-8', newline='') as stream:
                    has_rows = next(csv.DictReader(stream), None) is not None
                if has_rows:
                    plot_csv_metrics(csv_path, paths['figures_dir'])
                elif local_logger is not None:
                    local_logger.info(f'{split} CSV 只有表头，跳过曲线绘制：{csv_path}')

            run_summary = {
                'train_summary': train_summary,
                'experiment_dir': paths['experiment_dir'],
            }
            if config['evaluation']['test_after_train']:
                test_summary = run_test(
                    task=result['task'],
                    test_dataloader=result['dataloaders']['test'],
                    checkpoint_path=paths['output_paths']['checkpoints_dir'] / 'best.pt',
                    device=device,
                    metric_collection=result['metrics']['test'],
                    metrics_writer=result['writers']['test'],
                    logger=local_logger,
                )
                run_summary['test'] = test_summary
            return run_summary

        summary = context.run_main('训练后绘图与测试', postprocess)

        def finish_logger():
            if local_logger is not None:
                local_logger.finish(summary)
            return None

        context.run_local('记录运行结束', finish_logger)

    except Exception as error:
        primary_error = error
        raise

    finally:
        cleanup_errors = []
        try:
            if local_logger is not None:
                close_logger = getattr(local_logger, 'close', None)
                if callable(close_logger):
                    close_logger()
        except Exception as error:
            cleanup_errors.append(error)

        try:
            context.close()
        except Exception as error:
            cleanup_errors.append(error)

        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note(
                        f"清理运行资源时也发生异常：{type(error).__name__}: {error}"
                    )
            else:
                cleanup_failure = RuntimeError('运行清理失败喵!')
                for error in cleanup_errors:
                    cleanup_failure.add_note(f'{type(error).__name__}: {error}')
                raise cleanup_failure from cleanup_errors[0]

    return summary


if __name__ == '__main__':
    main()
