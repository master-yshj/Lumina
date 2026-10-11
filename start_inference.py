
from parse.parser import parse_config
from run_support import (parse_entry_args, make_cli_overrides, prepare_runtime,
                         prepare_run_outputs, seed_process)
from logger.logger import RunLogger
from tasks.registry import create_builder
from Inference.inference_runner import run_inference

try:
    import readline
except ImportError:
    pass

EXPECTED_MODE = 'inference'


def parse_args(argv=None):
    return parse_entry_args(EXPECTED_MODE, argv)


def main(argv=None):
    args = parse_args(argv)
    config = parse_config(args.config, cli_overrides=make_cli_overrides(args), expected_mode=EXPECTED_MODE)
    if args.check_config:
        print(f"配置解析与校验通过：task={config['task']['type']}，mode={config['run']['mode']}，未启动运行。")
        return config
    device, device_summary = prepare_runtime(config)
    paths = prepare_run_outputs(config)
    logger = RunLogger(paths['log_path'])
    logger.start({'任务': config['task']['type'], '运行模式': EXPECTED_MODE,
                  '配置路径': args.config, '设备': str(device), '实验目录': paths['experiment_dir']})
    seed_process(config['experiment']['seed'])
    builder = create_builder(config, device=device, device_summary=device_summary,
                             output_paths=paths['output_paths'], logger=logger)
    result = builder.build()
    loaded = run_inference(task=result['task'], task_inference=result['task_inference'],
                           checkpoint_path=config['run']['checkpoint']['path'],
                           device=device, logger=logger)
    completed = 0
    while True:
        try:
            input_data = result['task_inference'].read_input(input_fn=input)
            prediction = loaded['inferencer'].infer(input_data)
            print(prediction['result_text'])
            logger.info(f'prediction: {prediction}')

            print()
            print('=' * 60)
            print('模型回答：')
            print(prediction['result_text'])
            print('=' * 60)
            print()

            completed += 1
        except (ValueError, TypeError) as error:
            logger.info(f'本次输入无法推理，请检查输入内容：{error}')
        except (KeyboardInterrupt, EOFError) as error:
            reason = type(error).__name__
            break
    summary = dict(checkpoint_path=loaded['checkpoint_path'], completed_predictions=completed, exit_reason=reason)
    logger.finish(summary)
    return summary


if __name__ == '__main__':
    main()
