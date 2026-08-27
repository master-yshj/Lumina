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

LOSS_PARAMS_CUSTOM_DICT = {
    "crossentropyloss": {
        "ignore_index": -100,
    },
}

LOSS_ITEM_CUSTOM_DICT = {
    "name": "REQUIRED",
    "type": "REQUIRED",
    "weight": 1.0,
    "params": {},
}

METRIC_ITEM_CUSTOM_DICT = {
    "type": "REQUIRED",
    "params": {
        "ignore_index": -100,
    },
}

CUSTOM_DICT = {
    "name": "REQUIRED",
    "task": "REQUIRED",
    "checkpoint": {
        "path": "REQUIRED",
        "config_path": None,
    },
    "losses": [
        LOSS_ITEM_CUSTOM_DICT
    ],
    "dataset": {
        "type": "REQUIRED",
        "dataset_path": "REQUIRED",
        "test_path": "REQUIRED",
        "max_seq_length": 128,
        "mlm_prob": 0.15,
    },
    "dataloader": {
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
    },
    "cuda": {
        "device": "cpu",
        "gpu_nums": 1,
    },
    "metrics": {
        "test": [
            METRIC_ITEM_CUSTOM_DICT,
        ]
    }
}

def parse_args() -> Namespace:
    """
    用于解析命令行配置，并返回Namespace
    :param: None,
    :return: Namespace(config="cfg/example_qa.yaml",
                       resume="outputs/last.pt")
    """
    parser = ArgumentParser(description = "使用命令行运行程序，命令行示例：python start_train.py --config cfg/example_qa.yaml")
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

def check_and_add_config(config,template,loss_completed=False,key="config"):
    """
    用于规范化配置，进行异常值抛出和默认值填充
    :param config: dict or list or others
    :param template: dict or list or others
    :param loss_completed: bool，是否完成loss的硬编码处理
    :param key: config的键
    """

    # 首先硬编码处理loss
    if not loss_completed:
        loss_completed = True
        if not "losses" in config:
            raise ValueError(f"config中不存在losses这个键喵!")
        losses = config["losses"]
        if not isinstance(losses, list):
            raise ValueError(f"losses的类型必须为list喵! 它现在的type是: {type(losses)}")
        if len(losses) == 0:
            raise ValueError(f"losses不能为空喵!")
        for idx,loss in enumerate(losses):
            if not isinstance(loss,dict):
                raise ValueError(f"config.losses.loss 都必须为dict类型的喵!")
            if not "name" in loss:
                raise ValueError(f"config.losses[{idx}] 中没有找到name字段喵!")
            if not "type" in loss:
                raise ValueError(f"config.losses[{idx}] 中没有找到type字段喵!")
            if not "weight" in loss:
                raise ValueError(f"config.losses[{idx}] 中没有找到weight字段喵!")
            if not "params" in loss:
                raise ValueError(f"config.losses[{idx}] 中没有找到params字段喵!")
            name = loss["name"]
            loss_type = loss["type"]
            if not isinstance(loss_type,str):
                raise ValueError(f"config.losses{[idx]}.type 的值必须为字符串类型的喵!")
            if not loss_type.strip():
                raise ValueError(f"config.losses{[idx]}.type 的值不能为空喵!")
            loss_type = loss_type.casefold()
            weight = loss["weight"]
            params = loss["params"]
            allow_keys = {"name", "type", "weight", "params"}
            extra_keys = loss.keys() - allow_keys
            LOSS_PARAMS_CUSTOM_DICT_COPY = deepcopy(LOSS_PARAMS_CUSTOM_DICT)
            if extra_keys:
                raise ValueError(f"config.losses[{idx}] 中出现不合法字段喵! 合法字段有: {allow_keys}")
            if not name:
                raise ValueError(f"config.losses[{idx}].name 不能为空喵!")
            if not loss_type:
                raise ValueError(f"config.losses[{idx}].loss_type 不能为空喵!")
            if not loss_type in SUPPORTED_LOSSES:
                raise ValueError(f"config.losses[{idx}].loss_type: {loss_type} 不受支持喵!")
            if not loss_type in LOSS_PARAMS_CUSTOM_DICT_COPY:
                raise ValueError(f"config.losses[{idx}].loss_type: {loss_type} 不存在于通用模板LOSS_PARAMS_CUSTOM_DICT中!"
                                 f"LOSS_PARAMS_CUSTOM_DICT: {LOSS_PARAMS_CUSTOM_DICT}")
            if weight is None:
                config["losses"][idx]["weight"] = 1.0
            if params is None:
                raise ValueError(f"config.losses[{idx}].params的值不能为None喵!")
            if not isinstance(params,dict):
                raise ValueError(f"config.losses[{idx}].params的值必须为dict喵!")
            if params == {}:
                config["losses"][idx]["params"] = LOSS_PARAMS_CUSTOM_DICT_COPY[loss_type]
            else:
                for custom_key, custom_value in LOSS_PARAMS_CUSTOM_DICT_COPY[loss_type].items():
                    if custom_key not in params:
                        raise ValueError(f"config.losses[{idx}].params 中少了一个键 {custom_key} 喵!")
                for param_key, param_value in params.items():
                    if param_key not in LOSS_PARAMS_CUSTOM_DICT_COPY[loss_type]:
                        raise ValueError(f"config.losses[{idx}].params.{param_key} 这个键不应该存在喵!")
                    if param_value is None:
                        config["losses"][idx]["params"][param_key] = LOSS_PARAMS_CUSTOM_DICT_COPY[loss_type][param_key]
                    else:
                        pass

    if loss_completed:
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
                    raise ValueError(f"{key} 中缺少了一个键 {template_key} 喵! 请补上喵! 它的type是: {type(template_value)}")
                # 当template_value为REQUIRED字段时
                if template_value == "REQUIRED":
                    if config[template_key] is None:
                        raise ValueError(f"{template_key} 为必填字段喵!")
                if type(template_value) in (dict, list):
                    if key.startswith("config.metrics") and template_key == "params":
                        pass
                    elif config[template_key] is not None:
                        if isinstance(template_value, dict):
                            if isinstance(config[template_key], dict):
                                check_and_add_config(config[template_key],template_value,True,key+"."+template_key)
                            else:
                                raise ValueError(f"{config[template_key]} 的类型必须为dict喵! 它现在的type是: {type(config[template_key])}")
                        if isinstance(template_value, list):
                            if isinstance(config[template_key], list):
                                check_and_add_config(config[template_key],template_value,True,key+"."+template_key)
                            else:
                                raise ValueError(f"{config[template_key]} 的类型必须为list喵! 它现在的type是: {type(config[template_key])}")
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
                for index,config_elem in enumerate(config):
                    template_elem = template[0]
                    if template_elem == "REQUIRED":
                        if config[index] is None:
                            raise ValueError(f"config[{index}]: {config[index]} 为必填字段喵!")
                    if type(template_elem) in (dict, list):
                        if config_elem is not None:
                            if isinstance(template_elem, dict):
                                if isinstance(config_elem, dict):
                                    check_and_add_config(config_elem,template_elem,True,key+"["+str(index)+"]")
                                else:
                                    raise ValueError(f"{key} 的值 {config_elem} 的类型应该为dict喵! 它现在的type是: {type(config_elem)}")
                            if isinstance(template_elem, list):
                                if isinstance(config_elem, list):
                                    check_and_add_config(config_elem,template_elem,True,key+"["+str(index)+"]")
                                else:
                                    raise ValueError(f"{key} 的值 {config_elem} 的类型应该为list喵! 它现在的type是: {type(config_elem)}")
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
    dataset_path = pathlib.Path(config_dict["dataset"]["dataset_path"])
    test_path = pathlib.Path(config_dict["dataset"]["test_path"])
    checkpoint_path = pathlib.Path(config_dict["checkpoint"]["path"])
    config_path = config_dict["checkpoint"]["config_path"]
    if config_path is None:
        config_path = checkpoint_path.parent.parent / "config" / "config.yaml"
    config_path = pathlib.Path(config_path)
    test_path = dataset_path / test_path
    path_dict = {
        "dataset_path": dataset_path,
        "test_path": test_path,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
    }
    # 把里面的相对路径全部转化为绝对路径
    path_dict = relative_to_absolute_path_full_dict(path_dict,base_path)
    config_dict["dataset"]["dataset_path"] = path_dict["dataset_path"]
    config_dict["dataset"]["test_path"] = path_dict["test_path"]
    config_dict["checkpoint"]["path"] = path_dict["checkpoint_path"]
    config_dict["checkpoint"]["config_path"] = path_dict["config_path"]
    print("yaml有关路径的参数已经绝对化了喵~")
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

    # losses 字段
    losses = copy_dict["losses"]
    if not isinstance(losses, list):
        raise ValueError(f"config.losses: {losses} 必须为list类型的喵! 现在的type: {type(losses)}")
    if not losses:
        raise ValueError(f"config.losses 不能为空喵!")
    if not all(isinstance(loss, dict) for loss in losses):
        raise ValueError("config.losses 里面的元素都必须为dict类型的喵!")
    # losses 的每个 loss 字段的检查
    exist_name = []
    total_loss_weight = 0
    for idx, loss in enumerate(losses):
        copy_loss = deepcopy(loss)
        # loss.name 字段
        loss_name = copy_loss["name"]
        if not isinstance(loss_name, str):
            raise ValueError(
                f"config.losses[{idx}].name: {loss_name} 必须为字符串类型的喵! 现在的type: {type(loss_name)}")
        if not loss_name.strip():
            raise ValueError(f"config.losses[{idx}].name 的值不能为空喵!")
        if loss_name.strip().casefold() in exist_name:
            raise ValueError(f"config.losses[{idx}].name: {loss_name} 这一损失的名字在losses里面出现了至少两次喵!")
        exist_name.append(loss_name.strip().casefold())
        # loss.type 字段
        loss_type = copy_loss["type"]
        if not isinstance(loss_type, str):
            raise ValueError(f"config.losses[{idx}].type: {loss_type} 必须为字符串类型的喵!")
        if not loss_type.strip():
            raise ValueError(f"config.losses[{idx}].type 不能为空喵!")
        if not loss_type.strip().casefold() in SUPPORTED_LOSSES:
            raise ValueError(
                f"config.losses[{idx}].type: {loss_type} 这种类型的损失不受支持喵! 受支持的损失有: {SUPPORTED_LOSSES}")
        # loss.weight 字段
        loss_weight = copy_loss["weight"]
        if not type(loss_weight) in (int, float) or loss_weight < 0 or not isfinite(loss_weight):
            raise ValueError(f"config.losses[{idx}].weight: {loss_weight} 的值必须为非负数喵!")
        total_loss_weight += loss_weight
        # loss.params 子字段, 按type进行硬编码校验
        # losses.crossentropyloss
        if loss_type.strip().casefold() == "crossentropyloss":
            loss_params = copy_loss["params"]
            if not isinstance(loss_params, dict):
                raise ValueError(f"config.losses[{idx}].params: {loss_params} 的值必须为字典类型的喵!")
            if not loss_params:
                raise ValueError(f"config.losses[{idx}].params 不能为空喵!")
            if not "ignore_index" in loss_params:
                raise ValueError(f"config.losses[{idx}].params 里面缺少了 'ignore_index' 字段喵!")
            ignore_index = loss_params["ignore_index"]
            if not type(ignore_index) is int:
                raise ValueError(
                    f"config.losses[{idx}].params.ignore_index 必须为整型的喵! 现在的type: {type(ignore_index)}")
            if task.strip().casefold() == "mlm_pretrain":
                if ignore_index != -100:
                    raise ValueError(
                        f"config.losses[{idx}].params.ignore_index 在特定任务: config.task: {task} 中必须为-100喵!")
    if total_loss_weight == 0:
        raise ValueError(f"config.losses 中的所有loss.weight均为0喵! 无法进行训练喵!")

    # dataset 字段
    # dataset.type 字段
    dataset_type = copy_dict["dataset"]["type"]
    if not isinstance(dataset_type, str):
        raise ValueError(f"config.dataset.type: {dataset_type} 必须为字符串类型的喵!")
    if not dataset_type.strip():
        raise ValueError(f"config.dataset.type 不能为空喵!")
    if not dataset_type.casefold() in SUPPORTED_DATASET:
        raise ValueError(
            f"config.dataset.type: {dataset_type} 这种数据集类型不受支持喵! 受支持的数据集有: {SUPPORTED_DATASET}")
    # dataset.dataset_path 字段
    dataset_path = copy_dict["dataset"]["dataset_path"]
    if not isinstance(dataset_path, pathlib.Path):
        raise ValueError(
            f"config.dataset.dataset_path: {dataset_path} 必须为pathlib.Path类型的喵! 现在的type: {type(dataset_path)}")
    if not dataset_path.is_absolute():
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一路径没有绝对化喵!")
    if not dataset_path.exists():
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一路径不存在喵!")
    if not dataset_path.is_dir():
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一路径并未指向一个目录喵!")
    if not any(dataset_path.iterdir()):
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一目录下没有任何文件喵!")
    # dataset.test_path 字段
    test_path = copy_dict["dataset"]["test_path"]
    if not isinstance(test_path, pathlib.Path):
        raise ValueError(
            f"config.dataset.test_path: {test_path} 必须为pathlib.Path类型的喵! 现在的type: {type(test_path)}")
    if not test_path.is_absolute():
        raise ValueError(f"config.dataset.test_path: {test_path} 这一路径没有绝对化喵!")
    if not test_path.exists():
        raise ValueError(f"config.dataset.test_path: {test_path} 这一路径不存在喵!")
    if not test_path.is_file():
        raise ValueError(f"config.dataset.test_path: {test_path} 这一路径并未指向一个文件喵!")
    if test_path.stat().st_size == 0:
        raise ValueError(f"config.dataset.test_path: {test_path} 这一路径指向的文件大小为0字节喵!")
    with open(test_path, "r", encoding="utf-8") as test_file:
        if not any(line.strip() for line in test_file):
            raise ValueError(f"config.dataset.test_path: {test_path} 这一路径指向的文件为空喵!")
    # dataset.max_seq_length 字段
    max_seq_length = copy_dict["dataset"]["max_seq_length"]
    if not type(max_seq_length) is int or max_seq_length <= 0:
        raise ValueError(f"config.dataset.max_seq_length: {max_seq_length} 必须为正整数喵!")
    # dataset.mlm_prob 字段
    mlm_prob = copy_dict["dataset"]["mlm_prob"]
    if not type(mlm_prob) in (int, float) or not 0 < mlm_prob < 1 or not isfinite(mlm_prob):
        raise ValueError(f"config.dataset.mlm_prob: {mlm_prob} 必须为(0,1)之间的数字喵!")

    # dataloader 字段
    # dataloader.batch_size 字段
    batch_size = copy_dict["dataloader"]["batch_size"]
    if not type(batch_size) is int or batch_size <= 0:
        raise ValueError(f"config.dataloader.batch_size: {batch_size} 必须为正整数喵!")
    # dataloader.num_workers 字段
    num_workers = copy_dict["dataloader"]["num_workers"]
    if not type(num_workers) is int or num_workers < 0:
        raise ValueError(f"config.dataloader.num_workers: {num_workers} 必须为非负整数喵!")
    # dataloader.pin_memory 字段
    pin_memory = copy_dict["dataloader"]["pin_memory"]
    if not isinstance(pin_memory, bool):
        raise ValueError(f"config.dataloader.pin_memory: {pin_memory} 必须为bool类型的喵!")

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

    # metrics.test 字段
    metrics_test = copy_dict["metrics"]["test"]
    if not isinstance(metrics_test, list):
        raise ValueError(
            f"config.metrics.test: {metrics_test} 必须为list类型的喵! 现在的type: {type(metrics_test)}")
    if not metrics_test:
        raise ValueError(f"config.metrics.test 不能为空喵!")
    for idx, metric in enumerate(metrics_test):
        if not isinstance(metric, dict):
            raise ValueError(
                f"config.metrics.test[{idx}]: {metric} 必须为dict类型的喵! 现在的type: {type(metric)}")
        if not metric:
            raise ValueError(f"config.metrics.test[{idx}] 不能为空喵!")
        if "type" not in metric:
            raise ValueError(f"config.metrics.test[{idx}]: {metric} 缺少了type字段喵!")
        metric_type = metric["type"]
        if not isinstance(metric_type, str):
            raise ValueError(
                f"config.metrics.test[{idx}].type: {metric_type} 必须为字符串类型的喵! 现在的type: {type(metric_type)}")
        if not metric_type.strip():
            raise ValueError(f"config.metrics.test[{idx}].type 不能为空喵!")
        if metric_type.strip().casefold() not in SUPPORTED_METRICS:
            raise ValueError(
                f"config.metrics.test[{idx}].type: {metric_type} 这一metric类型不受支持喵! 受支持的metrics: {SUPPORTED_METRICS}")
        if "params" not in metric:
            raise ValueError(f"config.metrics.test[{idx}]: {metric} 缺少了params字段喵!")
        metric_params = metric["params"]
        if not isinstance(metric_params, dict):
            raise ValueError(
                f"config.metrics.test[{idx}].params: {metric_params} 必须为dict类型的喵! 现在的type: {type(metric_params)}")
        if not metric_params:
            raise ValueError(f"config.metrics.test[{idx}].params 不能为空喵!")
        # 根据metric的type进行硬编码校验
        # TokenAccuracy
        if metric_type.strip().casefold() == "tokenaccuracy":
            if "ignore_index" not in metric_params:
                raise ValueError(f"token_accuracy这一metric的params必须包含ignore_index字段喵!")
            ignore_index = metric_params["ignore_index"]
            if not type(ignore_index) is int:
                raise ValueError(f"ignore_index必须为整型的喵! 现在的type: {type(ignore_index)}")
            if not ignore_index == -100:
                raise ValueError(f"token_accuracy这一metric的ignore_index必须为-100喵! 现在是: {ignore_index}")
        # ExactMatch
        elif metric_type.strip().casefold() == "exact_match":
            if metric_params:
                raise ValueError(f"exact_match 这一metric的params必须为空喵!")
        # CharacterF1
        elif metric_type.strip().casefold() == "character_f1":
            if metric_params:
                raise ValueError(f"character_f1 这一metric的params必须为空喵!")

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