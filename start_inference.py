# PARSE_REFACTOR_DEMO: readline 为标准 input() 提供终端行编辑能力，支持
# Linux/WSL 中的方向键、历史记录和左右移动；导入失败时仍保留基础输入能力。
try:
    import readline
except ImportError:
    readline = None

from parse.parser import parse_args, parse_config
from start_train import EXPECTED_MODE
from tasks.registry import TASK_BUILDER_REGISTRY
from logger.logger import RunLogger
from experiments import create_inference_directories
from device import check_device
from Inference.inference_runner import run_inference

EXPECTED_MODE = "inference"

def main():
    args = parse_args()
    config_path = args.config
    config_dict, parse_context, compiled_schema = parse_config(
        config_path=config_path,
        expected_mode=EXPECTED_MODE,
        validate_runtime=True,
    )

    train_config_path = config_dict["run"]["checkpoint"]["train_config"]
    checkpoint_path = config_dict["run"]["checkpoint"]["path"]
    train_config_dir = train_config_path.parent
    experiment_dir = train_config_dir.parent
    inference_directories = create_inference_directories(experiment_dir)
    inference_dir = inference_directories["inference_dir"]
    inference_config_path = inference_directories["inference_config_path"]
    inference_logs_path = inference_directories["inference_logs_path"]

    logger = RunLogger(inference_logs_path)
    logger.start({
        "运行模式": "INFERENCE",
        "实验名称": config_dict["experiment"]["name"],
        "任务名称": config_dict["task"],
        "运行设备": config_dict["runtime"]["device"]["type"],
        "推理配置文件": config_path,
        "训练配置文件": train_config_path,
        "结果目录": inference_dir,
        "日志文件": inference_logs_path,
        "Checkpoint": config_dict["run"]["checkpoint"]["path"],
    })

    logger.info(f"推理即将开始喵!")

    device = config_dict["runtime"]["device"]["type"]
    gpu_nums = config_dict["runtime"]["device"]["count"]
    device_summary = check_device(device, gpu_nums)

    logger.info(f"当前设备信息: {device_summary}")

    task_type = config_dict["task"].strip().casefold()
    TASK_BUILDER_CANONICAL_MAP = TASK_BUILDER_REGISTRY.canonical_map
    builder = TASK_BUILDER_CANONICAL_MAP[task_type]
    task_components = builder(config_dict)
    task = task_components["task"]
    task_inference = task_components["task_inference"]

    run_summary = run_inference(
        task=task,
        task_inference=task_inference,
        checkpoint_path=checkpoint_path,
        device=device,
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
                result_text = prediction["result_text"]
                logger.info(f"Inference output: {prediction}")
                logger.info(f"{ '=' * 100}\n"
                            f"Result Text: {result_text}\n"
                            f"{ '=' * 100}\n")
            except ValueError as e:
                logger.info(f"本次处理失败了喵! {e}")
                continue
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