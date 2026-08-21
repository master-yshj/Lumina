import pathlib
import yaml
import sys
import re
import torch

from copy import deepcopy
from argparse import ArgumentParser,Namespace
from math import isfinite

# CUSTOM_DICT = {
#     'name': 'REQUIRED',
#     'model': {
#         'type': 'REQUIRED',
#         'model_path': 'REQUIRED',
#     },
#     'seed': 42,
#     'modules': {
#         'order': 'REQUIRED',
#         'layer_nums': 'REQUIRED'
#     },
#     'task': 'REQUIRED',
#     'cuda': {
#         'gpu_nums': 1,
#         'num_workers': 0,
#     },
#     'dataset_path': 'REQUIRED',
#     'train' : {
#         'dataset': {
#             'type': 'REQUIRED',
#             'dataset_train_path': 'REQUIRED',
#         },
#         'batch_size': 8,
#         'accumulation_steps': 1,
#         'total_iters': 10000,
#         'loss': {
#             'CrossEntropyLoss': 1.0,
#         },
#         'optimizer': 'AdamW',
#         'scheduler': 'StepLR',
#         "embedding": {
#             "type": "BertEmbedding",
#             "hidden_size": 128,
#             "max_position_embeddings": 128,
#             "type_vocab_size": 2,
#             "dropout": 0.1,
#             "layer_norm_eps": 1.0e-12,
#         },
#         'lr': 1.0e-3,
#     },
#     'valid': {
#             'dataset': {
#                 'type': 'REQUIRED',
#                 'dataset_val_path': 'REQUIRED',
#             },
#             'valid_iters': 500,
#             'early_stop_patience': 10,
#             'validation_rounds_before_save_ckpt': 1,
#         } ,
#     'test': {
#         'dataset': {
#             'type': 'REQUIRED',
#             'dataset_test_path': 'REQUIRED',
#         }
#     },
#     'tokenizer': {'type': 'ExampleTokenizer',
#                   'vocab_file_path': "../tokenizer/bert_base_chinese_tokenizer/vocab.txt"
#     },
# }

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
    "seed": 42,
    "task": "REQUIRED",
    "model": {
        "type": "REQUIRED",
        "name": "REQUIRED",
        "model_path": "REQUIRED",
        "hidden_size": 128,
        "max_position_embeddings": 128,
        "type_vocab_size": 2,
        "num_heads": 4,
        "intermediate_size": 512,
        "num_layers": 2,
        "embedding_dropout": 0.1,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "layer_norm_eps": 1.0e-12,
    },
    "tokenizer": {
        "type": "ExampleTokenizer",
        "vocab_file_path": "REQUIRED",
        "do_lower_case": True,
        "strip_accents": None,
        "max_input_chars_per_word": 100,
    },
    "dataset": {
        "type": "REQUIRED",
        "dataset_path": "REQUIRED",
        "train_path": "REQUIRED",
        "valid_path": "REQUIRED",
        "test_path": "REQUIRED",
        "max_seq_length": 128,
        "mlm_prob": 0.15,
    },
    "dataloader": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
    },
    "losses": [
        LOSS_ITEM_CUSTOM_DICT
    ],
    "optimizer": {
        "type": "AdamW",
        "lr": 1.0e-3,
        "weight_decay": 0.01,
        "betas":[0.9, 0.999],
        "eps": 1.0e-8,
    },
    "scheduler": {
        "type": "StepLR",
        "interval": "step",
        "step_size": 1000,
        "gamma": 0.1,
        "last_epoch": -1,
    },
    "train": {
        "total_iters": 10000,
        "accumulation_steps": 1,
        "log_every_iters": 100,
        "do_validation": True,
    },
    "valid": {
        "valid_every_iters": 500,
        "early_stop_patience": 10,
        "validation_rounds_before_save_ckpt": 1,
    },
    "test": {
        "enabled": False,
    },
    "cuda": {
        "device": "cpu",
        "gpu_nums": 1,
    },
    "metrics": {
        "train": [
            METRIC_ITEM_CUSTOM_DICT,
        ],
        "valid": [
            METRIC_ITEM_CUSTOM_DICT,
        ]
    }
}

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

def check_and_add_config(config,template,loss_completed=False,key="config"):
    """
    用于规范化配置，进行异常值抛出和默认值填充
    :param config: dict or list or others
    :param template: dict or list or others
    :param loss_completed: bool，是否完成loss的硬编码处理
    :param key: config的键
    """
    # CUSTOM_DICT_COPY = deepcopy(CUSTOM_DICT)
    # for dic_key,dic_value in dic.items():
    #     if dic_key not in CUSTOM_DICT_COPY:
    #         raise ValueError(f"你个大笨蛋！你是不是多写了一个键 {dic_key} ！请认真检查喵！")
    # for custom_key,custom_value in CUSTOM_DICT_COPY.items():
    #     if custom_key not in dic:
    #         if custom_value == 'REQUIRED':
    #             raise ValueError(f'你是不是忘了填参数 {custom_key} 的键了喵！请好好检查喵！')
    #         else :
    #             dic[custom_key] = CUSTOM_DICT_COPY[custom_key]
    #     if isinstance(custom_value,dict):
    #         if isinstance(dic[custom_key],dict):
    #             check_and_add_dict(dic[custom_key],CUSTOM_DICT_COPY[custom_key])
    #         else:
    #             raise ValueError(f"请把 {custom_key} 的值写成字典喵！")
    #     elif isinstance(custom_value,list):
    #         if isinstance(dic[custom_key],list):
    #             check_and_add_dict(dic[custom_key],CUSTOM_DICT_COPY[custom_key])
    #     if isinstance(custom_value,dict) and isinstance(dic[custom_key],dict) == False:
    #         raise ValueError(f"你 {custom_key} 的数据类型不是字典喵！传错了喵！")
    #     else:
    #         if dic[custom_key] is None and CUSTOM_DICT_COPY[custom_key] == 'REQUIRED':
    #             raise ValueError(f"你是不是忘了填键 {custom_key} 的值了喵！好好看看喵！")
    #         if dic[custom_key] is None and CUSTOM_DICT_COPY[custom_key] != 'REQUIRED':
    #             dic[custom_key] = CUSTOM_DICT_COPY[custom_key]
    #         else:
    #             pass

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
                    if config[template_key] is not None:
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
    model_path = pathlib.Path(config_dict["model"]["model_path"])
    vocab_file_path = pathlib.Path(config_dict["tokenizer"]["vocab_file_path"])
    dataset_path = pathlib.Path(config_dict["dataset"]["dataset_path"])
    train_path = pathlib.Path(config_dict["dataset"]["train_path"])
    valid_path = pathlib.Path(config_dict["dataset"]["valid_path"])
    test_path = pathlib.Path(config_dict["dataset"]["test_path"])
    train_path = dataset_path / train_path
    valid_path = dataset_path / valid_path
    test_path = dataset_path / test_path
    path_dict = {
        "model_path": model_path,
        "vocab_file_path": vocab_file_path,
        "dataset_path": dataset_path,
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
    }
    # 把里面的相对路径全部转化为绝对路径
    path_dict = relative_to_absolute_path_full_dict(path_dict,base_path)
    config_dict["model"]["model_path"] = path_dict["model_path"]
    config_dict["tokenizer"]["vocab_file_path"] = path_dict["vocab_file_path"]
    config_dict["dataset"]["dataset_path"] = path_dict["dataset_path"]
    config_dict["dataset"]["train_path"] = path_dict["train_path"]
    config_dict["dataset"]["valid_path"] = path_dict["valid_path"]
    config_dict["dataset"]["test_path"] = path_dict["test_path"]
    print("yaml有关路径的参数已经绝对化了喵~")
    # 将规范化后的配置返回
    return config_dict

def validate_config(config_dict):
    """
    对配置里面的参数进行合理性验证，使用硬编码
    :param config_dict:
    :return: correct_dict,dict:
    """
    # copy_dict = deepcopy(config_dict)
    # # name 字段
    # if not isinstance(copy_dict['name'],str):
    #     raise ValueError(f"任务名字 name: {copy_dict['name']} 必须是字符串类型的数据喵！")
    # if not copy_dict['name'].strip():
    #     raise ValueError(f"名字不能为空喵！")
    #
    # # model 字段
    # # model.type 字段
    # if not isinstance(copy_dict['model']['type'],str):
    #     raise ValueError(f"模型名字 model.type: {copy_dict['model']['type']} 必须是字符串类型的数据喵！")
    # if not copy_dict['model']['type'].strip():
    #     raise ValueError(f'模型名字不能为空喵！')
    # if copy_dict['model']['type'].casefold() not in SUPPORTED_MODEL_TYPE:
    #     raise ValueError(f"你真的确定 model.type: {copy_dict['model']['type']} 这个类型没有问题嘛，请检查相应的supported文件和目录喵~")
    # # model.model_path字段
    # if not copy_dict['model']['model_path'].is_dir():
    #     raise ValueError(f"该目录 model.model_path: {copy_dict['model']['model_path']} 不存在喵！请好好检查！")
    # else:
    #     is_empty = not any(copy_dict['model']['model_path'].iterdir())
    #     if not is_empty:
    #         pass
    #     else:
    #         raise ValueError(f"该目录 model.model_path: {copy_dict['model']['model_path']} 下面不存在任何文件喵！")
    #
    # # seed 字段
    # if not type(copy_dict['seed']) is int or copy_dict['seed'] <= 0:
    #     raise ValueError(f"种子必须要正整数喵！")
    #
    # # modules 字段
    # # modules.order 字段
    # if not isinstance(copy_dict['modules']['order'], list):
    #     raise ValueError(f"modules.order: {copy_dict['modules']['order']} 的值必须是列表类型的喵！")
    # if len(copy_dict['modules']['order']) == 0:
    #     raise ValueError(f"modules.order: {copy_dict['modules']['order']} 的列表内不能为空喵！")
    # if not all(isinstance(module, str) for module in copy_dict['modules']['order']):
    #     raise ValueError(f"modules.order: {copy_dict['modules']['order']} 里面的元素都必须要是字符串才行喵！")
    # if not all(module.strip() for module in copy_dict['modules']['order']):
    #     raise ValueError(f"modules.order: {copy_dict['modules']['order']} 里面的元素都不能为空喵！")
    # if not all(elem.casefold() in SUPPORTED_MODULES for elem in copy_dict['modules']['order']):
    #     raise ValueError(f"modules.order: {copy_dict['modules']['order']} 里面的元素不合法喵！找不到里面的模块喵！")
    # # modules.layer_nums 字段
    # if not isinstance(copy_dict['modules']['layer_nums'],list):
    #     raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 的值必须是列表类型的喵！")
    # if len(copy_dict['modules']['layer_nums']) == 0:
    #     raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 里面的值不能为空喵！")
    # if not all(type(module) is int for module in copy_dict['modules']['layer_nums']):
    #     raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 里面的元素都必须要是整型才行喵！")
    # if not all(num > 0 for num in copy_dict['modules']['layer_nums']):
    #     raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 里面的元素都必须是正整数喵！")
    # # modules.layer_nums和modules.order 比较字段
    # if not len(copy_dict['modules']['order']) == len(copy_dict['modules']['layer_nums']):
    #     raise ValueError(f"modules.order: {copy_dict['modules']['order']} 和 modules.layer_nums: {copy_dict['modules']['layer_nums']} 的列表层数不一样喵！请好好检查这两个参数喵！")
    #
    # # task 字段
    # if not isinstance(copy_dict['task'],str):
    #     raise ValueError(f"task: {copy_dict['task']} 的值必须是字符串类型的喵！")
    # if not copy_dict['task'].strip():
    #     raise ValueError(f"task: {copy_dict['task']} 的值不能为空喵！")
    # if not copy_dict['task'].casefold() in SUPPORTED_TASKS:
    #     raise ValueError(f"task: {copy_dict['task']} 的任务类型不受支持喵！")
    #
    # # cuda 字段
    # # cuda.gpu_nums 字段
    # if not type(copy_dict['cuda']['gpu_nums']) is int:
    #     raise ValueError(f"cuda.gpu_nums: {copy_dict['cuda']['gpu_nums']} 的值必须是整型的喵！")
    # if copy_dict['cuda']['gpu_nums'] <= 0:
    #     raise ValueError(f"cuda.gpu_nums: {copy_dict['cuda']['gpu_nums']} 的值必须是正整数的喵！0也不行喵！")
    # # cuda.num_workers 字段
    # if not type(copy_dict['cuda']['num_workers']) is int:
    #     raise ValueError(f"cuda.num_workers: {copy_dict['cuda']['num_workers']} 的值必须是整型的喵！")
    # if copy_dict['cuda']['num_workers'] < 0:
    #     raise ValueError(f"cuda.num_workers: {copy_dict['cuda']['num_workers']} 的值必须是正整数或者0才行的喵！")
    #
    # # dataset_path 字段
    # if not copy_dict['dataset_path'].is_dir():
    #     raise ValueError(
    #         f"dataset_path {copy_dict['dataset_path']} 不是目录喵！好好检查喵！")
    # else:
    #     is_empty = not any(copy_dict['dataset_path'].iterdir())
    #     if is_empty:
    #         raise ValueError(
    #             f"dataset_path: {copy_dict['dataset_path']} 该目录下面没有找到有效文件喵！")
    #     else:
    #         pass
    #
    # # train 字段
    # # train.dataset 字段
    # # train.dataset.type 字段
    # if not isinstance(copy_dict['train']['dataset']['type'],str):
    #     raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的值必须是字符串类型的喵！")
    # if not copy_dict['train']['dataset']['type'].strip():
    #     raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的值不能为空喵！")
    # if not copy_dict['train']['dataset']['type'].casefold() in SUPPORTED_DATASET:
    #     raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的数据集类型不受支持喵！")
    # # train.dataset.dataset_train_path 字段
    # if not copy_dict['train']['dataset']['dataset_train_path'].is_file():
    #     raise ValueError(f"train.dataset.dataset_train_path: {copy_dict['train']['dataset']['dataset_train_path']} 指向的不是一个文件喵！好好检查喵！")
    # if copy_dict['train']['dataset']['dataset_train_path'].stat().st_size == 0:
    #     raise ValueError(f"train.dataset.dataset_train_path {copy_dict['train']['dataset']['dataset_train_path']} 指向的文件是空的喵！")
    # with open(copy_dict['train']['dataset']['dataset_train_path'], 'r', encoding='utf-8') as f:
    #     if not any(line.strip() for line in f):
    #         raise ValueError(f"train.dataset.dataset_train_path {copy_dict['train']['dataset']['dataset_train_path']} 指向的文件是空的喵！")
    # # train.batch_size 字段
    # if not type(copy_dict['train']['batch_size']) is int or copy_dict['train']['batch_size'] <= 0:
    #     raise ValueError(f"train.batch_size: {copy_dict['train']['batch_size']} 的值必须为正整数喵！")
    # # train.accumulation_steps 字段
    # if not type(copy_dict['train']['accumulation_steps']) is int or copy_dict['train']['accumulation_steps'] <= 0:
    #     raise ValueError(f"train.accumulation_steps: {copy_dict['train']['accumulation_steps']} 的值必须为正整数喵！")
    # # train.total_iters 字段
    # if not type(copy_dict['train']['total_iters']) is int or copy_dict['train']['total_iters'] <= 0:
    #     raise ValueError(f"train.total_iters: {copy_dict['train']['total_iters']} 的值必须为正整数喵！")
    # # train.loss 字段
    # if not isinstance(copy_dict['train']['loss'], dict):
    #     raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的值必须为字典喵！")
    # if not copy_dict['train']['loss']:
    #     raise ValueError(f"train.loss: {copy_dict['train']['loss']} 里面是空的喵！")
    # for key,value in copy_dict['train']['loss'].items():
    #     if not isinstance(key, str):
    #         raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的键必须为字符串类型的喵！")
    #     if not key.strip():
    #         raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的键不能为空喵！")
    #     if (type(value) is float or type(value) is int) and value > 0:
    #         pass
    #     else:
    #         raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的值是损失的权重喵！必须为正数喵！")
    #     if not key.casefold() in SUPPORTED_LOSSES:
    #         raise ValueError(f"train.loss: {copy_dict['train']['loss']} 里面的 {key} 这种损失不受支持喵！请检查是否存在拼写错误（比如不能有下划线）")
    # # train.optimizer 字段
    # if not isinstance(copy_dict['train']['optimizer'], str):
    #     raise ValueError(f"train.optimizer: {copy_dict['train']['optimizer']} 的值必须为字符串类型的喵！")
    # if not copy_dict['train']['optimizer'].strip():
    #     raise ValueError(f"train.optimizer: {copy_dict['train']['optimizer']} 的值不能为空喵！")
    # if not copy_dict['train']['optimizer'].casefold() in SUPPORTED_OPTIMIZERS:
    #     raise ValueError(f"train.optimizer: {copy_dict['train']['optimizer']} 选择的优化器不受支持喵！")
    # # train.scheduler 字段
    # if not isinstance(copy_dict['train']['scheduler'], str):
    #     raise ValueError(f"train.scheduler: {copy_dict['train']['scheduler']} 的值必须为字符串类型的喵！")
    # if not copy_dict['train']['scheduler'].strip():
    #     raise ValueError(f"train.scheduler: {copy_dict['train']['scheduler']} 的值不能为空喵！")
    # if not copy_dict['train']['scheduler'].casefold() in SUPPORTED_SCHEDULERS:
    #     raise ValueError(f"train.scheduler: {copy_dict['train']['scheduler']} 选择的调度器不受支持喵！")
    # # train.embedding 字段
    # # train.embedding.type 字段
    # if not isinstance(copy_dict['train']['embedding']['type'], str):
    #     raise ValueError(f"train.embedding.type: {copy_dict['train']['embedding']['type']} 的值必须为字符串类型的喵！")
    # if not copy_dict['train']['embedding']['type'].strip():
    #     raise ValueError(f"train.embedding.type: {copy_dict['train']['embedding']['type']} 的值不能为空喵！")
    # if not copy_dict['train']['embedding']['type'].casefold() in SUPPORTED_EMBEDDINGS:
    #     raise ValueError(f"train.embedding.type: {copy_dict['train']['embedding']['type']} 选择的嵌入层不受支持喵！")
    # # train.embedding.hidden_size 字段
    # if not type(copy_dict['train']['embedding']['hidden_size']) is int or copy_dict['train']['embedding']['hidden_size'] <= 0:
    #     raise ValueError(f"train.embedding.hidden_size: {copy_dict['train']['embedding']['hidden_size']} 的值必须为正整数喵！")
    # # train.embedding.max_position_embeddings 字段
    # if not type(copy_dict['train']['embedding']['max_position_embeddings']) is int or copy_dict['train']['embedding']['max_position_embeddings'] <= 0:
    #     raise ValueError(f"train.embedding.max_position_embeddings: {copy_dict['train']['embedding']['max_position_embeddings']} 的值必须为正整数喵！")
    # # train.embedding.type_vocab_size 字段
    # if not type(copy_dict['train']['embedding']['type_vocab_size']) is int or copy_dict['train']['embedding']['type_vocab_size'] <= 0:
    #     raise ValueError(f"train.embedding.type_vocab_size: {copy_dict['train']['embedding']['type_vocab_size']} 的值必须为正整数喵！")
    # # train.embedding.dropout 字段
    # if not type(copy_dict['train']['embedding']['dropout']) in (int, float) or not 0 <= copy_dict['train']['embedding']['dropout'] < 1:
    #     raise ValueError(f"train.embedding.dropout: {copy_dict['train']['embedding']['dropout']} 的值必须为 [0,1) 之间的 int 或 float 喵！")
    # # train.embedding.layer_norm_eps 字段
    # if not type(copy_dict['train']['embedding']['layer_norm_eps']) in (int, float) or copy_dict['train']['embedding']['layer_norm_eps'] <= 0:
    #     raise ValueError(f"train.embedding.layer_norm_eps: {copy_dict['train']['embedding']['layer_norm_eps']} 的值必须为大于0的 int 或 float 喵！")
    # # train.lr 字段
    # if type(copy_dict['train']['lr']) not in (int,float) or copy_dict['train']['lr'] <= 0:
    #     raise ValueError(f"train.lr: {copy_dict['train']['lr']} 的值必须为正数喵！")
    #
    # # valid 字段
    # # valid.dataset 字段
    # # valid.dataset.type 字段
    # if not isinstance(copy_dict['valid']['dataset']['type'],str):
    #     raise ValueError(f"valid.dataset.type: {copy_dict['valid']['dataset']['type']} 的值必须为字符串类型的喵！")
    # if not copy_dict['valid']['dataset']['type'].strip():
    #     raise ValueError(f"valid.dataset.type: {copy_dict['valid']['dataset']['type']} 的值不能为空喵！")
    # if not copy_dict['valid']['dataset']['type'].casefold() in SUPPORTED_DATASET:
    #     raise ValueError(f"valid.dataset.type: {copy_dict['valid']['dataset']['type']} 这个数据集不受支持喵！")
    # # valid.dataset.dataset_val_path 字段
    # if not copy_dict['valid']['dataset']['dataset_val_path'].is_file():
    #     raise ValueError(f"valid.dataset.dataset_val_path: {copy_dict['valid']['dataset']['dataset_val_path']} 指向的不是一个文件喵！好好检查喵！")
    # if copy_dict['valid']['dataset']['dataset_val_path'].stat().st_size == 0:
    #     raise ValueError(f"valid.dataset.dataset_val_path {copy_dict['valid']['dataset']['dataset_val_path']} 指向的文件是空的喵！")
    # with open(copy_dict['valid']['dataset']['dataset_val_path'], 'r', encoding='utf-8') as f:
    #     if not any(line.strip() for line in f):
    #         raise ValueError(f"valid.dataset.dataset_val_path {copy_dict['valid']['dataset']['dataset_val_path']} 指向的文件是空的喵！")
    # # valid.valid_iters 字段
    # if not type(copy_dict['valid']['valid_iters']) is int or copy_dict['valid']['valid_iters'] <= 0:
    #     raise ValueError(f"valid.valid_iters: {copy_dict['valid']['valid_iters']} 的值必须要为正整数喵！")
    # # valid.early_stop_patience
    # if not type(copy_dict['valid']['early_stop_patience']) is int or copy_dict['valid']['early_stop_patience'] <= 0:
    #     raise ValueError(f"valid.early_stop_patience: {copy_dict['valid']['early_stop_patience']} 的值必须要为正整数喵！")
    # # valid.validation_rounds_before_save_ckpt 字段
    # if not type(copy_dict['valid']['validation_rounds_before_save_ckpt']) is int or copy_dict['valid']['validation_rounds_before_save_ckpt'] <= 0:
    #     raise ValueError(f"valid.validation_rounds_before_save_ckpt: {copy_dict['valid']['validation_rounds_before_save_ckpt']} 的值必须为正整数喵！")
    #
    # # test 字段
    # # test.dataset 字段
    # # test.dataset.type 字段
    # if not isinstance(copy_dict['test']['dataset']['type'],str):
    #     raise ValueError(f"test.dataset.type: {copy_dict['test']['dataset']['type']} 的值必须为字符串类型的喵！")
    # if not copy_dict['test']['dataset']['type'].strip():
    #     raise ValueError(f"test.dataset.type: {copy_dict['test']['dataset']['type']} 的值不能为空喵！")
    # if not copy_dict['test']['dataset']['type'].casefold() in SUPPORTED_DATASET:
    #     raise ValueError(f"test.dataset.type: {copy_dict['test']['dataset']['type']} 这个数据集不受支持喵！")
    # # test.dataset.dataset_test_path 字段
    # if not copy_dict['test']['dataset']['dataset_test_path'].is_file():
    #     raise ValueError(f"test.dataset.dataset_test_path: {copy_dict['test']['dataset']['dataset_test_path']} 指向的不是一个文件喵！好好检查喵！")
    # if copy_dict['test']['dataset']['dataset_test_path'].stat().st_size == 0:
    #     raise ValueError(f"test.dataset.dataset_test_path {copy_dict['test']['dataset']['dataset_test_path']} 指向的文件是空的喵！")
    # with open(copy_dict['test']['dataset']['dataset_test_path'], 'r', encoding='utf-8') as f:
    #     if not any(line.strip() for line in f):
    #         raise ValueError(f"test.dataset.dataset_test_path {copy_dict['test']['dataset']['dataset_test_path']} 指向的文件是空的喵！")
    #
    # # tokenizer 字段
    # # tokenizer.type 字段
    # if not isinstance(copy_dict['tokenizer']['type'], str):
    #     raise ValueError(f"tokenizer.type: {copy_dict['tokenizer']['type']} 的值必须为字符串类型的喵！")
    # if not copy_dict['tokenizer']['type'].strip():
    #     raise ValueError(f"tokenizer.type: {copy_dict['tokenizer']['type']} 的值不能为空喵！")
    # if not copy_dict['tokenizer']['type'].casefold() in SUPPORTED_TOKENIZERS:
    #     raise ValueError(f"tokenizer.type: {copy_dict['tokenizer']['type']} 这种tokenizer不受支持喵！")
    # # tokenizer.vocab_file_path 字段
    # if not copy_dict['tokenizer']['vocab_file_path'].is_file():
    #     raise ValueError(f"tokenizer.vocab_file_path: {copy_dict['tokenizer']['vocab_file_path']} 指向的不是一个文件喵！请好好检查喵！")
    # if copy_dict['tokenizer']['vocab_file_path'].stat().st_size == 0:
    #     raise ValueError(f"tokenizer.vocab_file_path: {copy_dict['tokenizer']['vocab_file_path']} 指向的文件是空的喵！")
    # with open(copy_dict['tokenizer']['vocab_file_path'], 'r', encoding='utf-8') as f:
    #     if not any(line.strip() for line in f):
    #         raise ValueError(f"tokenizer.vocab_file_path: {copy_dict['tokenizer']['vocab_file_path']} 指向的文件是空的喵！")

    copy_dict = deepcopy(config_dict)

    # name 字段
    name = copy_dict["name"]
    if not isinstance(name, str):
        raise ValueError(f"config.name: {name} 必须要为字符串类型的喵! 现在的type: {type(name)}")
    if not name.strip():
        raise ValueError(f"config.name 不能为空喵!")

    # seed 字段
    seed = copy_dict["seed"]
    if not type(seed) is int:
        raise ValueError(f"config.seed: {seed} 必须为整型类型的喵! 现在的type: {type(seed)}")
    if seed < 0:
        raise ValueError(f"config.seed: {seed} 必须为0或者正整数喵!")

    # task 字段
    task = copy_dict["task"]
    if not isinstance(task, str):
        raise ValueError(f"config.task: {task} 必须为字符串类型的喵! 现在的type: {type(task)}")
    if not task.strip():
        raise ValueError(f"config.task 不能为空喵!")
    if not task.casefold() in SUPPORTED_TASKS:
        raise ValueError(f"config.task: {task} 这种任务类型不受支持喵! 支持的任务类型: {SUPPORTED_TASKS}")

    # model 字段
    # model.type 字段
    model_type = copy_dict["model"]["type"]
    if not isinstance(model_type, str):
        raise ValueError(f"config.model.type: {model_type} 必须为字符串类型的喵! 现在的type: {type(model_type)}")
    if not model_type.strip():
        raise ValueError(f"config.model.type 不能为空喵!")
    if not model_type.casefold() in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"config.model.type: {model_type} 这种模型族类型不受支持喵! 支持的模型族类型: {SUPPORTED_MODEL_TYPES}")
    # model.name 字段
    model_name = copy_dict["model"]["name"]
    if not isinstance(model_name, str):
        raise ValueError(f"config.model.name: {model_name} 必须为字符串类型的喵! 现在的type: {type(model_name)}")
    if not model_name.strip():
        raise ValueError(f"config.model.name 不能为空喵!")
    if not model_name.casefold() in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"config.model.name: {model_name} 这种模型类型不受支持喵! 支持的模型有: {SUPPORTED_MODEL_NAMES}")
    # model.model_path 字段
    model_path = copy_dict["model"]["model_path"]
    if not isinstance(model_path, pathlib.Path):
        raise ValueError(f"config.model.model_path: {model_path} 必须为pathlib.Path类型的喵! 现在的type: {type(model_path)}")
    if not model_path.is_absolute():
        raise ValueError(f"config.model.model_path: {model_path} 还没有绝对化喵!")
    if not model_path.exists():
        raise ValueError(f"config.model.model_path: {model_path} 该路径不存在喵!")
    if not model_path.is_dir():
        raise ValueError(f"config.model.model_path: {model_path} 该路径并未指向一个文件夹喵!")
    if not any(model_path.iterdir()):
        raise ValueError(f"config.model.model_path: {model_path} 该目录下并未包含任何文件喵!")
    # model.hidden_size 字段
    hidden_size = copy_dict["model"]["hidden_size"]
    if not type(hidden_size) is int:
        raise ValueError(f"config.model.hidden_size: {hidden_size} 必须为整型类型的喵! 现在的type: {type(hidden_size)}")
    if hidden_size <= 0:
        raise ValueError(f"config.model.hidden_size: {hidden_size} 必须为正整数喵!")
    # model.max_position_embeddings 字段
    max_position_embeddings = copy_dict["model"]["max_position_embeddings"]
    if not type(max_position_embeddings) is int:
        raise ValueError(f"config.model.max_position_embeddings: {max_position_embeddings} 必须为整型类型的喵! 现在的type: {type(max_position_embeddings)}")
    if max_position_embeddings <= 0:
        raise ValueError(f"config.model.max_position_embeddings: {max_position_embeddings} 必须为正整数喵!")
    # model.type_vocab_size 字段
    type_vocab_size = copy_dict["model"]["type_vocab_size"]
    if not type(type_vocab_size) is int:
        raise ValueError(f"config.model.type_vocab_size: {type_vocab_size} 必须为整型类型的喵! 现在的type: {type(type_vocab_size)}")
    if type_vocab_size <= 0:
        raise ValueError(f"config.model.type_vocab_size: {type_vocab_size} 必须为正整数喵!")
    # model.num_heads 字段
    num_heads = copy_dict["model"]["num_heads"]
    if not type(num_heads) is int:
        raise ValueError(f"config.model.num_heads: {num_heads} 必须为整型类型的喵! 现在的type: {type(num_heads)}")
    if num_heads <= 0:
        raise ValueError(f"config.model.num_heads: {num_heads} 必须为正整数喵!")
    # model.intermediate_size 字段
    intermediate_size = copy_dict["model"]["intermediate_size"]
    if not type(intermediate_size) is int:
        raise ValueError(f"config.model.intermediate_size: {intermediate_size} 必须为整型类型的喵! 现在的type: {type(intermediate_size)}")
    if intermediate_size <= 0:
        raise ValueError(f"config.model.intermediate_size: {intermediate_size} 必须为正整数喵!")
    # model.num_layers 字段
    num_layers = copy_dict["model"]["num_layers"]
    if not type(num_layers) is int:
        raise ValueError(f"config.model.num_layers: {num_layers} 必须为整型类型的喵! 现在的type: {type(num_layers)}")
    if num_layers <= 0:
        raise ValueError(f"config.model.num_layers: {num_layers} 必须为正整数喵!")
    # model.embedding_dropout 字段
    embedding_dropout = copy_dict["model"]["embedding_dropout"]
    if not type(embedding_dropout) in (float, int) or not 0<= embedding_dropout < 1 or not isfinite(embedding_dropout):
        raise ValueError(f"config.model.embedding_dropout: {embedding_dropout} 的值必须为[0,1)的浮点数或者整数喵!")
    # model.attention_dropout 字段
    attention_dropout = copy_dict["model"]["attention_dropout"]
    if not type(attention_dropout) in (float, int) or not 0 <= attention_dropout < 1 or not isfinite(attention_dropout):
        raise ValueError(f"config.model.attention_dropout: {attention_dropout} 的值必须为[0,1)的浮点数或者整数喵!")
    # model.hidden_dropout 字段
    hidden_dropout = copy_dict["model"]["hidden_dropout"]
    if not type(hidden_dropout) in (float, int) or not 0 <= hidden_dropout < 1 or not isfinite(hidden_dropout):
        raise ValueError(f"config.model.hidden_dropout: {hidden_dropout} 的值必须为[0,1)的浮点数或者整数喵!")
    # model.layer_norm_eps 字段
    layer_norm_eps = copy_dict["model"]["layer_norm_eps"]
    if not type(layer_norm_eps) in (float, int) or not layer_norm_eps > 0 or not isfinite(layer_norm_eps):
        raise ValueError(f"config.model.layer_norm_eps: {layer_norm_eps} 的值必须为有限正数喵!")
    # hidden_size 与 num_heads 的跨字段约束
    if not hidden_size % num_heads == 0:
        raise ValueError(f"config.model.hidden_size: {hidden_size}\n"
                         f"config.model.num_heads: {num_heads}\n"
                         f"hidden_size 必须要能够被 num_heads 整除喵!\n")

    # tokenizer 字段
    # tokenizer.type 字段
    tokenizer_type = copy_dict["tokenizer"]["type"]
    if not isinstance(tokenizer_type, str):
        raise ValueError(f"config.tokenizer.type: {tokenizer_type} 的值必须为字符串类型的喵! 现在的type: {type(tokenizer_type)}")
    if not tokenizer_type.strip():
        raise ValueError(f"config.tokenizer.type 的值不能为空喵!")
    if not tokenizer_type.casefold() in SUPPORTED_TOKENIZERS:
        raise ValueError(f"config.tokenizer.type: {tokenizer_type} 这一分词器不受支持喵! 支持的分词器有: {SUPPORTED_TOKENIZERS}")
    # tokenizer.vocab_file_path 字段
    vocab_file_path = copy_dict["tokenizer"]["vocab_file_path"]
    if not isinstance(vocab_file_path, pathlib.Path):
        raise ValueError(f"config.tokenizer.vocab_file_path: {vocab_file_path} 的值必须为pathlib.Path类型的喵! 现在的type: {type(vocab_file_path)}")
    if not vocab_file_path.is_absolute():
        raise ValueError(f"config.tokenizer.vocab_file_path: {vocab_file_path} 该路径还没有绝对化喵!")
    if not vocab_file_path.exists():
        raise ValueError(f"config.tokenizer.vocab_file_path: {vocab_file_path} 该路径不存在喵!")
    if not vocab_file_path.is_file():
        raise ValueError(f"config.tokenizer.vocab_file_path: {vocab_file_path} 该路径没有指向一个文件喵!")
    if vocab_file_path.stat().st_size == 0:
        raise ValueError(f"config.tokenizer.vocab_file_path: {vocab_file_path} 指向的文件大小为0字节的喵!")
    with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
        if not any(line.strip() for line in vocab_file):
            raise ValueError(f"config.tokenizer.vocab_file_path: {vocab_file_path} 指向的文件为空喵!")
    # tokenizer.do_lower_case 字段
    do_lower_case = copy_dict["tokenizer"]["do_lower_case"]
    if not isinstance(do_lower_case, bool):
        raise ValueError(f"config.tokenizer.do_lower_case: {do_lower_case} 的值必须为bool类型的喵! 当前的type: {type(do_lower_case)}")
    # tokenizer.strip_accents 字段
    strip_accents = copy_dict["tokenizer"]["strip_accents"]
    if not type(strip_accents) in (bool, type(None)):
        raise ValueError(f"config.tokenizer.strip_accents: {strip_accents} 的值必须为bool类型或者None喵! 当前的type: {type(strip_accents)}")
    # tokenizer.max_input_chars_per_word 字段
    max_input_chars_per_word = copy_dict["tokenizer"]["max_input_chars_per_word"]
    if not type(max_input_chars_per_word) is int or max_input_chars_per_word <= 0:
        raise ValueError(f"config.tokenizer.max_input_chars_per_word: {max_input_chars_per_word} 的值必须为正整数喵!")

    # dataset 字段
    # dataset.type 字段
    dataset_type = copy_dict["dataset"]["type"]
    if not isinstance(dataset_type, str):
        raise ValueError(f"config.dataset.type: {dataset_type} 必须为字符串类型的喵!")
    if not dataset_type.strip():
        raise ValueError(f"config.dataset.type 不能为空喵!")
    if not dataset_type.casefold() in SUPPORTED_DATASET:
        raise ValueError(f"config.dataset.type: {dataset_type} 这种数据集类型不受支持喵! 受支持的数据集有: {SUPPORTED_DATASET}")
    # dataset.dataset_path 字段
    dataset_path = copy_dict["dataset"]["dataset_path"]
    if not isinstance(dataset_path, pathlib.Path):
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 必须为pathlib.Path类型的喵! 现在的type: {type(dataset_path)}")
    if not dataset_path.is_absolute():
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一路径没有绝对化喵!")
    if not dataset_path.exists():
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一路径不存在喵!")
    if not dataset_path.is_dir():
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一路径并未指向一个目录喵!")
    if not any(dataset_path.iterdir()):
        raise ValueError(f"config.dataset.dataset_path: {dataset_path} 这一目录下没有任何文件喵!")
    # dataset.train_path 字段
    train_path = copy_dict["dataset"]["train_path"]
    if not isinstance(train_path, pathlib.Path):
        raise ValueError(f"config.dataset.train_path: {train_path} 必须为pathlib.Path类型的喵! 现在的type: {type(train_path)}")
    if not train_path.is_absolute():
        raise ValueError(f"config.dataset.train_path: {train_path} 这一路径没有绝对化喵!")
    if not train_path.exists():
        raise ValueError(f"config.dataset.train_path: {train_path} 这一路径不存在喵!")
    if not train_path.is_file():
        raise ValueError(f"config.dataset.train_path: {train_path} 这一路径并未指向一个文件喵!")
    if train_path.stat().st_size == 0:
        raise ValueError(f"config.dataset.train_path: {train_path} 这一路径指向的文件大小为0字节喵!")
    with open(train_path, "r", encoding="utf-8") as train_file:
        if not any(line.strip() for line in train_file):
            raise ValueError(f"config.dataset.train_path: {train_path} 这一路径指向的文件为空喵!")
    # dataset.valid_path 字段
    valid_path = copy_dict["dataset"]["valid_path"]
    if not isinstance(valid_path, pathlib.Path):
        raise ValueError(f"config.dataset.valid_path: {valid_path} 必须为pathlib.Path类型的喵! 现在的type: {type(valid_path)}")
    if not valid_path.is_absolute():
        raise ValueError(f"config.dataset.valid_path: {valid_path} 这一路径没有绝对化喵!")
    if not valid_path.exists():
        raise ValueError(f"config.dataset.valid_path: {valid_path} 这一路径不存在喵!")
    if not valid_path.is_file():
        raise ValueError(f"config.dataset.valid_path: {valid_path} 这一路径并未指向一个文件喵!")
    if valid_path.stat().st_size == 0:
        raise ValueError(f"config.dataset.valid_path: {valid_path} 这一路径指向的文件大小为0字节喵!")
    with open(valid_path, "r", encoding="utf-8") as valid_file:
        if not any(line.strip() for line in valid_file):
            raise ValueError(f"config.dataset.valid_path: {valid_path} 这一路径指向的文件为空喵!")
    # dataset.test_path 字段
    test_path = copy_dict["dataset"]["test_path"]
    if not isinstance(test_path, pathlib.Path):
        raise ValueError(f"config.dataset.test_path: {test_path} 必须为pathlib.Path类型的喵! 现在的type: {type(test_path)}")
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
    # dataset.max_seq_length 与 model.max_position_embeddings 的跨字段约束
    if max_seq_length > max_position_embeddings:
        raise ValueError(f"config.dataset.max_seq_length: {max_seq_length}\n"
                         f"config.model.max_position_embeddings: {max_position_embeddings}\n"
                         f"max_seq_length 的值必须小于等于 max_position_embeddings喵!\n")
    # dataset.mlm_prob 字段
    mlm_prob = copy_dict["dataset"]["mlm_prob"]
    if not type(mlm_prob) in (int, float) or not 0 < mlm_prob < 1 or not isfinite(mlm_prob):
        raise ValueError(f"config.dataset.mlm_prob: {mlm_prob} 必须为(0,1)之间的数字喵!")

    # dataloader 字段
    # dataloader.batch_size 字段
    batch_size = copy_dict["dataloader"]["batch_size"]
    if not type(batch_size) is int or batch_size <= 0:
        raise ValueError(f"config.dataloader.batch_size: {batch_size} 必须为正整数喵!")
    # dataloader.shuffle 字段
    shuffle = copy_dict["dataloader"]["shuffle"]
    if not isinstance(shuffle, bool):
        raise ValueError(f"config.dataloader.shuffle: {shuffle} 必须为bool类型的喵!")
    # dataloader.num_workers 字段
    num_workers = copy_dict["dataloader"]["num_workers"]
    if not type(num_workers) is int or num_workers < 0:
        raise ValueError(f"config.dataloader.num_workers: {num_workers} 必须为非负整数喵!")
    # dataloader.pin_memory 字段
    pin_memory = copy_dict["dataloader"]["pin_memory"]
    if not isinstance(pin_memory, bool):
        raise ValueError(f"config.dataloader.pin_memory: {pin_memory} 必须为bool类型的喵!")
    # dataloader.drop_last 字段
    drop_last = copy_dict["dataloader"]["drop_last"]
    if not isinstance(drop_last, bool):
        raise ValueError(f"config.dataloader.drop_last: {drop_last} 必须为bool类型的喵!")
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
            raise ValueError(f"config.losses[{idx}].name: {loss_name} 必须为字符串类型的喵! 现在的type: {type(loss_name)}")
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
            raise ValueError(f"config.losses[{idx}].type: {loss_type} 这种类型的损失不受支持喵! 受支持的损失有: {SUPPORTED_LOSSES}")
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
                raise ValueError(f"config.losses[{idx}].params.ignore_index 必须为整型的喵! 现在的type: {type(ignore_index)}")
            if task.strip().casefold() == "mlm_pretrain":
                if ignore_index != -100:
                    raise ValueError(f"config.losses[{idx}].params.ignore_index 在特定任务: config.task: {task} 中必须为-100喵!")
    if total_loss_weight == 0:
        raise ValueError(f"config.losses 中的所有loss.weight均为0喵! 无法进行训练喵!")

    # optimizer 字段
    # optimizer.type 字段
    optimizer_type = copy_dict["optimizer"]["type"]
    if not isinstance(optimizer_type, str):
        raise ValueError(f"config.optimizer.type: {optimizer_type} 必须为字符串类型的喵!")
    if not optimizer_type.strip():
        raise ValueError(f"config.optimizer.type 不能为空喵!")
    if not optimizer_type.casefold() in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"config.optimizer.type: {optimizer_type} 这种优化器类型不受支持喵!")
    # optimizer.lr 字段
    lr = copy_dict["optimizer"]["lr"]
    if not type(lr) in (int, float) or lr <= 0 or not isfinite(lr):
        raise ValueError(f"config.optimizer.lr: {lr} 必须为正数喵!")
    # optimizer.weight_decay 字段
    weight_decay = copy_dict["optimizer"]["weight_decay"]
    if not type(weight_decay) in (int, float) or weight_decay < 0 or not isfinite(weight_decay):
        raise ValueError(f"config.optimizer.weight_decay: {weight_decay} 必须为非负数喵!")
    # optimizer.betas 字段
    betas = copy_dict["optimizer"]["betas"]
    if not isinstance(betas, list):
        raise ValueError(f"config.optimizer.betas: {betas} 必须为list类型的喵!")
    if not len(betas) == 2:
        raise ValueError(f"config.optimizer.betas: {betas} 的长度必须为2喵! 现在的长度: {len(betas)}")
    for idx, beta in enumerate(betas):
        if not type(beta) in (int, float) or not 0 <= beta < 1 or not isfinite(beta):
            raise ValueError(f"config.optimizer.betas[{idx}]: {beta} 的值必须为[0,1)之间的数值喵!")
    # optimizer.eps 字段
    optimizer_eps = copy_dict["optimizer"]["eps"]
    if not type(optimizer_eps) in (float, int) or optimizer_eps <= 0 or not isfinite(optimizer_eps):
        raise ValueError(f"config.optimizer.eps: {optimizer_eps} 必须为有限正数喵!")

    # scheduler 字段
    # scheduler.type 字段
    scheduler_type = copy_dict["scheduler"]["type"]
    if not isinstance(scheduler_type, str):
        raise ValueError(f"config.scheduler.type: {scheduler_type} 必须为字符串类型的喵! 现在的type: {type(scheduler_type)}")
    if not scheduler_type.strip():
        raise ValueError(f"config.scheduler.type 不能为空喵!")
    if not scheduler_type.casefold() in SUPPORTED_SCHEDULERS:
        raise ValueError(f"config.scheduler.type: {scheduler_type} 这种类型的调度器不受支持喵! 支持的调度器有: {SUPPORTED_SCHEDULERS}")
    # scheduler.interval 字段
    interval = copy_dict["scheduler"]["interval"]
    if not isinstance(interval, str):
        raise ValueError(f"config.scheduler.interval: {interval} 必须为字符串类型的喵! 现在的type: {type(interval)}")
    if not interval.strip():
        raise ValueError(f"config.scheduler.interval 不能为空喵!")
    if not interval.strip().casefold() == 'step':
        raise ValueError(f"config.scheduler.interval: {interval} 的值不受支持喵! 受支持的interval有: {'step'}")
    # scheduler.step_size 字段
    step_size = copy_dict["scheduler"]["step_size"]
    if not type(step_size) is int or step_size <= 0 or not isfinite(step_size):
        raise ValueError(f"config.scheduler.step_size: {step_size} 必须为正整数喵!")
    # scheduler.gamma 字段
    gamma = copy_dict["scheduler"]["gamma"]
    if not type(gamma) in (int, float) or gamma < 0 or not isfinite(gamma):
        raise ValueError(f"config.scheduler.gamma: {gamma} 必须为非负数值喵!")
    # scheduler.last_epoch
    last_epoch = copy_dict["scheduler"]["last_epoch"]
    if not type(last_epoch) is int or last_epoch < -1:
        raise ValueError(f"config.scheduler.last_epoch: {last_epoch} 必须为大于等于-1的整数喵!")

    # train 字段
    # train.total_iters 字段
    total_iters = copy_dict["train"]["total_iters"]
    if not type(total_iters) is int or total_iters <= 0 or not isfinite(total_iters):
        raise ValueError(f"config.train.total_iters: {total_iters} 必须为正整数喵!")
    # train.accumulation_steps 字段
    accumulation_steps = copy_dict["train"]["accumulation_steps"]
    if not type(accumulation_steps) is int or accumulation_steps <= 0 or not isfinite(accumulation_steps):
        raise ValueError(f"config.train.accumulation_steps: {accumulation_steps} 必须为正整数喵!")
    # train.log_every_iters 字段
    log_every_iters = copy_dict["train"]["log_every_iters"]
    if not type(log_every_iters) is int or log_every_iters <= 0 or not isfinite(log_every_iters):
        raise ValueError(f"config.train.log_every_iters: {log_every_iters} 必须为正整数喵!")
    # train.do_validation 字段
    do_validation = copy_dict["train"]["do_validation"]
    if not isinstance(do_validation, bool):
        raise ValueError(f"config.train.do_validation: {do_validation} 必须为bool类型的喵!")
    # total_iters 与 log_every_iters 的跨字段约束
    if log_every_iters > total_iters:
        raise ValueError(f"config.train.total_iters: {total_iters}\n"
                         f"config.train.log_every_iters: {log_every_iters}\n"
                         f"total_iters 的值必须大于等于 log_every_iters 的值喵!\n")

    # valid 字段
    # valid.valid_every_iters 字段
    valid_every_iters = copy_dict["valid"]["valid_every_iters"]
    if not type(valid_every_iters) is int or valid_every_iters <= 0 or not isfinite(valid_every_iters):
        raise ValueError(f"config.valid.valid_every_iters: {valid_every_iters} 必须为正整数喵!")
    # valid_every_iters 与 total_iters 的跨字段约束
    if do_validation:
        if valid_every_iters > total_iters:
            raise ValueError(f"config.train.total_iters: {total_iters}\n"
                             f"config.valid.valid_every_iters: {valid_every_iters}\n"
                             f"total_iters 的值必须大于 valid_every_iters 喵!\n")
    # valid.early_stop_patience 字段
    early_stop_patience = copy_dict["valid"]["early_stop_patience"]
    if not type(early_stop_patience) is int or early_stop_patience <= 0 or not isfinite(early_stop_patience):
        raise ValueError(f"config.valid.early_stop_patience: {early_stop_patience} 必须为正整数喵!")
    # valid.validation_rounds_before_save_ckpt 字段
    validation_rounds_before_save_ckpt = copy_dict["valid"]["validation_rounds_before_save_ckpt"]
    if not type(validation_rounds_before_save_ckpt) is int or validation_rounds_before_save_ckpt <= 0 or not isfinite(validation_rounds_before_save_ckpt):
        raise ValueError(f"config.valid.validation_rounds_before_save_ckpt: {validation_rounds_before_save_ckpt} 必须为正整数喵!")

    # test 字段
    # test.enabled 字段
    test_enabled = copy_dict["test"]["enabled"]
    if not isinstance(test_enabled, bool):
        raise ValueError(f"config.test.enabled: {test_enabled} 必须为bool类型的喵!")
    if test_enabled:
        pass
    else:
        pass

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
                raise ValueError(f"config.cuda.device 中填的数值{cuda_n} 超过了本机实际的cuda数: {torch.cuda.device_count()} 喵!")
    # cuda.gpu_nums 字段
    gpu_nums = copy_dict["cuda"]["gpu_nums"]
    if not type(gpu_nums) is int or gpu_nums <= 0 or not isfinite(gpu_nums):
        raise ValueError(f"config.cuda.gpu_nums: {gpu_nums} 必须为正整数喵!")

    # metrics 字段
    # metrics.train 字段
    metrics_train = copy_dict["metrics"]["train"]
    if not isinstance(metrics_train, list):
        raise ValueError(f"config.metrics.train: {metrics_train} 必须为list类型的喵! 现在的type: {type(metrics_train)}")
    if not metrics_train:
        raise ValueError(f"config.metrics.train 不能为空喵!")
    for idx, metric in enumerate(metrics_train):
        if not isinstance(metric, dict):
            raise ValueError(f"config.metrics.train[{idx}]: {metric} 必须为dict类型的喵! 现在的type: {type(metric)}")
        if not metric:
            raise ValueError(f"config.metrics.train[{idx}] 不能为空喵!")
        if "type" not in metric:
            raise ValueError(f"config.metrics.train[{idx}]: {metric} 缺少了type字段喵!")
        metric_type = metric["type"]
        if not isinstance(metric_type, str):
            raise ValueError(f"config.metrics.train[{idx}].type: {metric_type} 必须为字符串类型的喵! 现在的type: {type(metric_type)}")
        if not metric_type.strip():
            raise ValueError(f"config.metrics.train[{idx}].type 不能为空喵!")
        if metric_type.strip().casefold() not in SUPPORTED_METRICS:
            raise ValueError(f"config.metrics.train[{idx}].type: {metric_type} 这一metric类型不受支持喵! 受支持的metrics: {SUPPORTED_METRICS}")
        if "params" not in metric:
            raise ValueError(f"config.metrics.train[{idx}]: {metric} 缺少了params字段喵!")
        metric_params = metric["params"]
        if not isinstance(metric_params, dict):
            raise ValueError(f"config.metrics.train[{idx}].params: {metric_params} 必须为dict类型的喵! 现在的type: {type(metric_params)}")
        if not metric_params:
            raise ValueError(f"config.metrics.train[{idx}].params 不能为空喵!")
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
    # metrics.valid 字段
    metrics_valid = copy_dict["metrics"]["valid"]
    if not isinstance(metrics_valid, list):
        raise ValueError(f"config.metrics.valid: {metrics_valid} 必须为list类型的喵! 现在的type: {type(metrics_valid)}")
    if not metrics_valid:
        raise ValueError(f"config.metrics.valid 不能为空喵!")
    for idx, metric in enumerate(metrics_valid):
        if not isinstance(metric, dict):
            raise ValueError(f"config.metrics.valid[{idx}]: {metric} 必须为dict类型的喵! 现在的type: {type(metric)}")
        if not metric:
            raise ValueError(f"config.metrics.valid[{idx}] 不能为空喵!")
        if "type" not in metric:
            raise ValueError(f"config.metrics.valid[{idx}]: {metric} 缺少了type字段喵!")
        metric_type = metric["type"]
        if not isinstance(metric_type, str):
            raise ValueError(f"config.metrics.valid[{idx}].type: {metric_type} 必须为字符串类型的喵! 现在的type: {type(metric_type)}")
        if not metric_type.strip():
            raise ValueError(f"config.metrics.valid[{idx}].type 不能为空喵!")
        if metric_type.strip().casefold() not in SUPPORTED_METRICS:
            raise ValueError(f"config.metrics.valid[{idx}].type: {metric_type} 这一metric类型不受支持喵! 受支持的metrics: {SUPPORTED_METRICS}")
        if "params" not in metric:
            raise ValueError(f"config.metrics.valid[{idx}]: {metric} 缺少了params字段喵!")
        metric_params = metric["params"]
        if not isinstance(metric_params, dict):
            raise ValueError(f"config.metrics.valid[{idx}].params: {metric_params} 必须为dict类型的喵! 现在的type: {type(metric_params)}")
        if not metric_params:
            raise ValueError(f"config.metrics.valid[{idx}].params 不能为空喵!")
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

def normalize_resume_path(resume):
    """
    将控制台传入的resume参数规范化
    :param resume: resume参数，模型开始训练的ckpt保存的pt文件
    :return:
    """
    if resume is None:
        return None
    if not type(resume) in (str, pathlib.Path):
        raise ValueError(f"resume 必须为字符串类型或 pathlib.Path 类型的喵! 现在的type: {type(resume)}")
    if isinstance(resume, str):
        if not resume.strip():
            raise ValueError(f"resume 为字符串类型时不能为空喵!")
    resume_path = pathlib.Path(resume).resolve()
    if not resume_path.exists():
        raise ValueError(F"resume_path: {resume_path} 指向的路径不存在喵!")
    if not resume_path.is_file():
        raise ValueError(f"resume_path: {resume_path} 必须指向文件喵!")
    if resume_path.suffix.casefold() == ".pt":
        return resume_path
    else:
        raise ValueError(f"resume_path: {resume_path} 必须指向pt类型文件喵!")


