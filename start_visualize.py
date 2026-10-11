
from parse.parser import parse_config
from run_support import (parse_entry_args, make_cli_overrides, prepare_runtime,
                         prepare_run_outputs, seed_process)
from logger.logger import RunLogger
from tasks.registry import create_builder
from run_support import load_visualization_task_weights

EXPECTED_MODE = 'visualize'


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
    checkpoint_path = load_visualization_task_weights(result['task'], config['run']['checkpoint']['path'])
    result['task'].to(device)
    result['task'].eval()
    completed = []
    for item in result['visualizers']:
        item['visualizer'].visualize(model=result['model'], tokenizer=result['tokenizer'],
                                     dataloader=result['dataloaders']['visualize'],
                                     save_dir=paths['visualize_dir'] / item['name'],
                                     device=device, logger=logger)
        completed.append(item['name'])
    summary = dict(checkpoint_path=checkpoint_path, methods=completed, visualize_dir=paths['visualize_dir'])
    logger.finish(summary)
    return summary


if __name__ == '__main__':
    main()
