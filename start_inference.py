from parse.inference_parser import parse_args, load_config
from tasks.registry import TASK_BUILDER_REGISTRY
from logger.logger import RunLogger
from experiments import create_inference_directories
from device import check_device
from Inference.inference_runner import run_inference


def main():
    args = parse_args()
    config_path = args.config
    config_dict = load_config(config_path)

    train_config_path = config_dict["checkpoint"]["config_path"]
    checkpoint_path = config_dict["checkpoint"]["path"]
    train_config_dir = train_config_path.parent
    experiment_dir = train_config_dir.parent
    inference_directories = create_inference_directories(experiment_dir)
    inference_dir = inference_directories["inference_dir"]
    inference_config_path = inference_directories["inference_config_path"]
    inference_logs_path = inference_directories["inference_logs_path"]

    logger = RunLogger(inference_logs_path)
    logger.start({
        "运行模式": "INFERENCE",
        "实验名称": config_dict["name"],
        "任务名称": config_dict["task"],
        "运行设备": config_dict["cuda"]["device"],
        "推理配置文件": config_path,
        "训练配置文件": train_config_path,
        "结果目录": inference_dir,
        "日志文件": inference_logs_path,
        "Checkpoint": config_dict["checkpoint"]["path"],
    })

    logger.info(f"推理即将开始喵!")

    device = config_dict["cuda"]["device"]
    gpu_nums = config_dict["cuda"]["gpu_nums"]
    device_summary = check_device(device, gpu_nums)

    logger.info(f"当前设备信息: {device_summary}")

    task_type = config_dict["task"].strip().casefold()
    builder = TASK_BUILDER_REGISTRY[task_type]
    task_components = builder(config_dict)
    task = task_components["task"]
    task_inference = task_components["task_inference"]

    run_summary = run_inference(
        task=task,
        task_inference=task_inference,
        checkpoint_path=config_dict["checkpoint"]["path"],
        device=config_dict["cuda"]["device"],
        logger=logger,
    )

    inferencer = run_summary["inferencer"]
    checkpoint_path = run_summary["checkpoint_path"]

    completed_inputs = 0
    exit_reason = "未知原因"

    try:
        while True:
            text = input("请输入文本:")
            try:
                prediction = inferencer.infer(text)
            except ValueError as e:
                logger.info(f"本次处理失败了喵! {e}")
                continue
            logger.info(f"predictions: {prediction}")
            completed_inputs += 1
    except KeyboardInterrupt:
        exit_reason = "用户通过 Ctrl+C 退出"
    except EOFError:
        exit_reason = "输入流结束"
    finally:
        logger.finish({
            "退出原因": exit_reason,
            "成功处理数量": completed_inputs,
            "Checkpoint": checkpoint_path,
        })

if __name__ == "__main__":
    main()