import pathlib
import yaml
import sys
import re
import torch

from copy import deepcopy
from argparse import ArgumentParser,Namespace
from math import isfinite

# 构建根目录以导包
ROOT_PATH = pathlib.Path(__file__).parent.parent
MODELS_PATH = ROOT_PATH / 'models'
TASKS_PATH = ROOT_PATH / 'tasks'

ROOT_STR = str(ROOT_PATH)
if ROOT_STR not in sys.path:
    sys.path.append(ROOT_STR)

try:
    import models.supported
    from models.supported import (SUPPORTED_MODEL_TYPES,
                                  SUPPORTED_MODEL_NAMES,
                                  SUPPORTED_LOSSES,
                                  SUPPORTED_TOKENIZERS,
                                  SUPPORTED_OPTIMIZERS,
                                  SUPPORTED_SCHEDULERS,)

    import tasks.registry
    from tasks.registry import (SUPPORTED_TASKS, )

    import dataset
    from dataset.supported import (SUPPORTED_DATASET,)

    import metrics
    from metrics.registry import (SUPPORTED_METRICS, )

finally:
    if ROOT_STR in sys.path:
        sys.path.remove(ROOT_STR)

INFERENCE_PARAMS_CUSTOM_DICT = {
    "mlm_pretrain_inference": {
        "top_k": 5,
        "max_seq_length": 128,
    },
}

CUSTOM_DICT = {
    "name": "REQUIRED",
    "task": "REQUIRED",
    "checkpoint": {
        "path": "REQUIRED",
        "config_path": None,
    },
    "cuda": {
        "device": "cpu",
        "gpu_nums": 1,
    },
    "inference": None,
}

def parse_args() -> Namespace:
    """
    用于解析命令行配置，并返回Namespace
    :param: None,
    :return: Namespace(config="cfg/example.yaml",
                       resume="outputs/last.pt")
    """
    parser = ArgumentParser(description = "使用命令行运行程序，命令行示例：python start_train.py --config cfg/example.yaml")
    # 配置config参数
    parser.add_argument("--config",help = "要运行哪一个任务，读取相应的配置",type=str,required=True)
    # 配置ckpt参数，以便继续训练，键名叫resume
    parser.add_argument("--resume",help="要运行哪一个任务的检查点，从检查点开始运行，读取相应的配置",type=str,default=None)
    return parser.parse_args()

def load_yaml(yaml_path):
    """
    加载yaml配置，输入yaml的路径，返回解析yaml的字典
    :param yaml_path:
    :return: yaml_dict, dict;
    """
    path = pathlib.Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML 配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        yaml_dict = yaml.safe_load(file)
    if yaml_dict is None:
        raise ValueError(f"YAML 配置文件为空：{path}")
    if not isinstance(yaml_dict, dict):
        raise ValueError("YAML 顶层必须是键值映射")
    return yaml_dict

def check_and_add_config(config,template,key="config"):
    """
    用于规范化配置，进行异常值抛出和默认值填充
    :param config: dict or list or others
    :param template: dict or list or others
    :param key: config的键
    """

    # 当template为dict时
    if isinstance(template, dict):
        if not isinstance(config, dict):
            raise ValueError(f"{key}: 的类型必须为dict喵!")
        for dic_key, dic_value in config.items():
            if dic_key not in template:
                raise ValueError(f"{dic_key} 这个键不应该存在于 {key} 中喵! 请好好检查喵!")
        for template_key, template_value in template.items():
            if template_key == "losses":
                continue
            if template_key not in config:
                raise ValueError(
                    f"{key} 中缺少了一个键 {template_key} 喵! 请补上喵! 它的type是: {type(template_value)}")
            # 当template_value为REQUIRED字段时
            if template_value == "REQUIRED":
                if config[template_key] is None:
                    raise ValueError(f"{template_key} 为必填字段喵!")
            if type(template_value) in (dict, list):
                if config[template_key] is not None:
                    if isinstance(template_value, dict):
                        if isinstance(config[template_key], dict):
                            check_and_add_config(config[template_key], template_value,key + "." + template_key)
                        else:
                            raise ValueError(
                                f"{config[template_key]} 的类型必须为dict喵! 它现在的type是: {type(config[template_key])}")
                    if isinstance(template_value, list):
                        if isinstance(config[template_key], list):
                            check_and_add_config(config[template_key], template_value, key + "." + template_key)
                        else:
                            raise ValueError(
                                f"{config[template_key]} 的类型必须为list喵! 它现在的type是: {type(config[template_key])}")
                else:
                    raise ValueError(f"{template_key} 的值不能为空喵!")
            else:
                if config[template_key] is None:
                    config[template_key] = template_value
                else:
                    pass

    # 当template为list时
    if isinstance(template, list):
        if not isinstance(config, list):
            raise ValueError(f"{key} 的类型必须为list喵!")
        if len(config) == 0:
            raise ValueError(f"{key} 的值不能为空喵!")
        else:
            for index, config_elem in enumerate(config):
                template_elem = template[0]
                if template_elem == "REQUIRED":
                    if config[index] is None:
                        raise ValueError(f"config[{index}]: {config[index]} 为必填字段喵!")
                if type(template_elem) in (dict, list):
                    if config_elem is not None:
                        if isinstance(template_elem, dict):
                            if isinstance(config_elem, dict):
                                check_and_add_config(config_elem, template_elem, key + "[" + str(index) + "]")
                            else:
                                raise ValueError(
                                    f"{key} 的值 {config_elem} 的类型应该为dict喵! 它现在的type是: {type(config_elem)}")
                        if isinstance(template_elem, list):
                            if isinstance(config_elem, list):
                                check_and_add_config(config_elem, template_elem, key + "[" + str(index) + "]")
                            else:
                                raise ValueError(
                                    f"{key} 的值 {config_elem} 的类型应该为list喵! 它现在的type是: {type(config_elem)}")
                    else:
                        raise ValueError(f"{key} 的值 (如{template_elem}) 不能为空喵!")
                else:
                    if config_elem is None:
                        config[index] = template_elem
                    else:
                        pass

def relative_to_absolute_path(relative_path,base_path):
    """
    将基础路径和相对路径拼接为绝对路径，便于进一步修改配置中的数据
    :param relative_path: 相对路径
    :param base_path: 基础路径
    :return: absolute_path 绝对路径
    """
    relative_path = pathlib.Path(relative_path)
    base_path = pathlib.Path(base_path)
    if pathlib.Path(relative_path).is_absolute():
        return relative_path.resolve()
    else:
        absolute_path = base_path / relative_path
        return absolute_path.resolve()

def relative_to_absolute_path_full_dict(path_dict,base_path):
    for key,value in path_dict.items():
        value = pathlib.Path(value)
        path_dict[key] = relative_to_absolute_path(value,base_path)
    return path_dict

def build_config(raw_dict, yaml_path):
    """
    将python字典构建为config参数
    :param raw_dict: 从load_yaml里面拿到的python字典
    :param yaml_path: 从parse_args输出的Namespace拿到的config参数，也就是yaml路径
    :return:
    """
    config_dict = deepcopy(raw_dict)
    # 校验字典参数并使用默认值填充未填写的值
    check_and_add_config(config_dict,CUSTOM_DICT)
    print("yaml字典已经规范化了喵~")
    # 获取yaml所在的目录
    base_path = pathlib.Path(yaml_path).resolve().parent
    checkpoint_path = pathlib.Path(config_dict["checkpoint"]["path"])
    config_path = config_dict["checkpoint"]["config_path"]
    if config_path is None:
        config_path = checkpoint_path.parent.parent / "config" / "config.yaml"
    config_path = pathlib.Path(config_path)
    path_dict = {
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
    }
    # 把里面的相对路径全部转化为绝对路径
    path_dict = relative_to_absolute_path_full_dict(path_dict,base_path)
    config_dict["checkpoint"]["path"] = path_dict["checkpoint_path"]
    config_dict["checkpoint"]["config_path"] = path_dict["config_path"]
    print("yaml有关路径的参数已经绝对化了喵~")
    inference_dict = config_dict["inference"]
    task = config_dict["task"]
    task = task.strip().casefold()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"config.task: {task} 这种任务类型不受支持喵! 支持的任务类型: {SUPPORTED_TASKS}")
    if inference_dict is None:
        config_dict["inference"] = INFERENCE_PARAMS_CUSTOM_DICT[task]
    else:
        for inference_key, inference_value in inference_dict.items():
            if inference_key in INFERENCE_PARAMS_CUSTOM_DICT[task] and inference_value is None:
                config_dict["inference"][inference_key] = INFERENCE_PARAMS_CUSTOM_DICT[task][inference_key]
            if not inference_key in INFERENCE_PARAMS_CUSTOM_DICT[task]:
                raise ValueError(f"config.inference 中出现了非法字段喵: {inference_key}")
        for custom_key, custom_value in INFERENCE_PARAMS_CUSTOM_DICT[task].items():
            if custom_key not in inference_dict:
                config_dict["inference"][custom_key] = custom_value
    print("yaml有关inference的参数已经默认填充了喵~")
    # 将规范化后的配置返回
    return config_dict

def validate_config(config_dict):
    """
    对配置里面的参数进行合理性验证，使用硬编码
    :param config_dict:
    :return: correct_dict,dict:
    """
    copy_dict = deepcopy(config_dict)
    # name 字段
    name = copy_dict["name"]
    if not isinstance(name, str):
        raise ValueError(f"config.name: {name} 必须要为字符串类型的喵! 现在的type: {type(name)}")
    if not name.strip():
        raise ValueError(f"config.name 不能为空喵!")

    # task 字段
    task = copy_dict["task"]
    if not isinstance(task, str):
        raise ValueError(f"config.task: {task} 必须为字符串类型的喵! 现在的type: {type(task)}")
    if not task.strip():
        raise ValueError(f"config.task 不能为空喵!")
    if not task.casefold() in SUPPORTED_TASKS:
        raise ValueError(f"config.task: {task} 这种任务类型不受支持喵! 支持的任务类型: {SUPPORTED_TASKS}")

    # checkpoint 字段
    # checkpoint.path 字段
    checkpoint_path = copy_dict["checkpoint"]["path"]
    if not isinstance(checkpoint_path, pathlib.Path):
        raise ValueError(f"config.checkpoint.path: {checkpoint_path} 必须为pathlib.Path类型的喵! 现在的type: {type(checkpoint_path)}")
    if not checkpoint_path.is_absolute():
        raise ValueError(f"config.checkpoint.path: {checkpoint_path} 现在还没有绝对化喵!")
    if not checkpoint_path.suffix.casefold() == ".pt":
        raise ValueError(f"config.checkpoint.path: {checkpoint_path} 必须指向一个pt文件喵!")
    # checkpoint.config_path 字段
    config_path = copy_dict["checkpoint"]["config_path"]
    if not isinstance(config_path, pathlib.Path):
        raise ValueError(f"config.checkpoint.config_path: {config_path} 必须为pathlib.Path类型的喵! 现在的type: {type(config_path)}")
    if not config_path.is_absolute():
        raise ValueError(f"config.checkpoint.config_path: {config_path} 现在还没有绝对化喵!")
    if not config_path.suffix.casefold() == ".yaml":
        raise ValueError(f"config.checkpoint.config_path: {config_path} 必须指向一个yaml文件喵!")
    if not config_path.exists():
        raise ValueError(f"config.checkpoint.config_path: {config_path} 该路径不存在喵!")
    if not config_path.is_file():
        raise ValueError(f"config.checkpoint.config_path: {config_path} 该路径并未指向一个文件喵!")

    # cuda 字段
    # cuda.device 字段
    cuda_device = copy_dict["cuda"]["device"]
    if not isinstance(cuda_device, str):
        raise ValueError(f"config.cuda.device: {cuda_device} 必须为字符串类型的喵!")
    if not cuda_device.strip():
        raise ValueError(f"config.cuda.device 不能为空喵!")
    device_pattern = r'(cpu|mps|cuda|cuda:\d+)$'
    if not re.fullmatch(device_pattern, cuda_device):
        raise ValueError(f"config.cuda.device: {cuda_device} 只能为cpu,cuda,cuda:N(N为十进制数字) 喵!")
    device_pattern_except_cpu = r'(cuda|cuda:(\d+))$'
    device_pattern_cuda_n = r'(cuda:(\d+))$'
    if re.fullmatch(device_pattern_except_cpu, cuda_device):
        if not torch.cuda.is_available():
            raise ValueError(f"未检测到本机有可用cuda  喵!")
        if re.fullmatch(device_pattern_cuda_n, cuda_device):
            res = re.match(device_pattern_cuda_n, cuda_device)
            cuda_n = int(res.group(2))
            if cuda_n >= torch.cuda.device_count():
                raise ValueError(
                    f"config.cuda.device 中填的数值{cuda_n} 超过了本机实际的cuda数: {torch.cuda.device_count()} 喵!")
    # cuda.gpu_nums 字段
    gpu_nums = copy_dict["cuda"]["gpu_nums"]
    if not type(gpu_nums) is int or gpu_nums <= 0 or not isfinite(gpu_nums):
        raise ValueError(f"config.cuda.gpu_nums: {gpu_nums} 必须为正整数喵!")

    # inference 字段
    inference_dict = copy_dict["inference"]
    if not isinstance(inference_dict, dict):
        raise ValueError(f"config.inference: {inference_dict} 必须为dict类型的喵! 现在的type: {type(inference_dict)}")
    if not inference_dict:
        raise ValueError(f"config.inference: {inference_dict} 不能为空喵!")
    # 根据task类型进行硬编码校验
    # mlm_pretrain_inference 字段
    if task.strip().casefold() == "mlm_pretrain_inference":
        if "top_k" not in inference_dict:
            raise ValueError(f"config.inference: {inference_dict} 需要包含 top_k 字段喵!")
        if "max_seq_length" not in inference_dict:
            raise ValueError(f"config.inference: {inference_dict} 需要包含 max_seq_length 字段喵!")
        top_k = inference_dict["top_k"]
        if not type(top_k) is int or top_k <= 0 or not isfinite(top_k):
            raise ValueError(f"config.inference.top_k: {top_k} 必须为有限正整数喵!")
        max_seq_length = inference_dict["max_seq_length"]
        if not type(max_seq_length) is int or max_seq_length <= 0 or not isfinite(max_seq_length):
            raise ValueError(f"config.inference.max_seq_length: {max_seq_length} 必须为有限正整数喵!")

def load_config(yaml_path):
    """
    加载配置，通过yaml路径读取yaml，创建config，并对其校验
    :param yaml_path:
    :return: config_dict:
    """
    yaml_dict = load_yaml(yaml_path)
    config_dict = build_config(yaml_dict, yaml_path)
    validate_config(config_dict)
    return config_dict