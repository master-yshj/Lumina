import pathlib
import yaml
import sys

from copy import deepcopy
from argparse import ArgumentParser,Namespace

CUSTOM_DICT = {
    'name': 'REQUIRED',
    'model': {
        'type': 'REQUIRED',
        'model_path': 'REQUIRED',
    },
    'seed': 42,
    'modules': {
        'order': 'REQUIRED',
        'layer_nums': 'REQUIRED'
    },
    'task': 'REQUIRED',
    'cuda': {
        'gpu_nums': 1,
        'num_workers': 0,
    },
    'train' : {
        'dataset': {
            'type': 'REQUIRED',
            'dataset_path': 'REQUIRED',
        },
        'batch_size': 8,
        'accumulation_steps': 1,
        'early_stop_patience': 10,
        'total_iters': 10000,
        'loss': {
            'cross_entropy_loss': 1.0,
        },
        'optimizer': 'AdamW',
        'scheduler': 'StepLR',
        'lr': 1e-3,
    },
    'tokenizer': {'type': 'ExampleTokenizer',
                  'vocab_file_path': "../tokenizer/bert_base_chinese_tokenizer/vocab.txt"
    },
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
    from models.supported import (SUPPORTED_MODEL_TYPE,
                                  SUPPORTED_MODULES,
                                  SUPPORTED_LOSSES,
                                  SUPPORTED_TOKENIZERS)

    import tasks.supported
    from tasks.supported import (SUPPORTED_TASKS,)

    import dataset
    from dataset.supported import (SUPPORTED_DATASET,)

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

def check_and_add_dict(dic,CUSTOM_DICT):
    """
    用于规范化字典，进行异常值抛出和默认值填充
    :param dic: 从load_yaml里面拿到的python原始字典
    :return:
    """
    CUSTOM_DICT_COPY = deepcopy(CUSTOM_DICT)
    for dic_key,dic_value in dic.items():
        if dic_key not in CUSTOM_DICT_COPY:
            raise ValueError(f"你个大笨蛋！你是不是多写了一个键 {dic_key} ！请认真检查喵！")
    for custom_key,custom_value in CUSTOM_DICT_COPY.items():
        if custom_key not in dic:
            if custom_value == 'REQUIRED':
                raise ValueError(f'你是不是忘了填参数 {custom_key} 的键了喵！请好好检查喵！')
            else :
                dic[custom_key] = CUSTOM_DICT_COPY[custom_key]
        if isinstance(custom_value,dict):
            if isinstance(dic[custom_key],dict):
                check_and_add_dict(dic[custom_key],CUSTOM_DICT_COPY[custom_key])
            else:
                raise ValueError(f"请把 {custom_key} 的值写成字典喵！")
        if isinstance(custom_value,dict) and isinstance(dic[custom_key],dict) == False:
            raise ValueError(f"你 {custom_key} 的数据类型不是字典喵！传错了喵！")
        else:
            if dic[custom_key] is None and CUSTOM_DICT_COPY[custom_key] == 'REQUIRED':
                raise ValueError(f"你是不是忘了填键 {custom_key} 的值了喵！好好看看喵！")
            if dic[custom_key] is None and CUSTOM_DICT_COPY[custom_key] != 'REQUIRED':
                dic[custom_key] = CUSTOM_DICT_COPY[custom_key]
            else:
                pass

def relative_to_absolute_path(relative_path,base_path):
    """
    将基础路径和相对路径拼接为绝对路径，便于进一步修改配置中的数据
    :param relative_path: 相对路径
    :param base_path: 基础路径
    :return: absolute_path 绝对路径
    """
    if pathlib.Path(relative_path).is_absolute():
        return relative_path.resolve()
    else:
        absolute_path = base_path / relative_path
        return absolute_path.resolve()

def relative_to_absolute_path_full_dict(dic,base_path):
    path_list = ["model_path","dataset_path","vocab_file_path"]
    for key,value in dic.items():
        if isinstance(value,dict):
            relative_to_absolute_path_full_dict(value,base_path)
        else:
            if key in path_list:
                dic[key] = relative_to_absolute_path(value,base_path)
            else:
                pass

def build_config(raw_dict, yaml_path):
    """
    将python字典构建为config参数
    :param raw_dict: 从load_yaml里面拿到的python字典
    :param yaml_path: 从parse_args输出的Namespace拿到的config参数，也就是yaml路径
    :return:
    """
    config_dict = deepcopy(raw_dict)
    # 校验字典参数并使用默认值填充未填写的值
    check_and_add_dict(config_dict,CUSTOM_DICT)
    print("yaml字典已经规范化了喵~")
    # 获取yaml所在的目录
    base_path = pathlib.Path(yaml_path).resolve().parent
    # 把里面的相对路径全部转化为绝对路径
    relative_to_absolute_path_full_dict(config_dict,base_path)
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
    if not isinstance(copy_dict['name'],str):
        raise ValueError(f"任务名字 name: {copy_dict['name']} 必须是字符串类型的数据喵！")
    if not copy_dict['name'].strip():
        raise ValueError(f"名字不能为空喵！")
    # model 字段
    # model.type 字段
    if not isinstance(copy_dict['model']['type'],str):
        raise ValueError(f"模型名字 model.type: {copy_dict['model']['type']} 必须是字符串类型的数据喵！")
    if not copy_dict['model']['type'].strip():
        raise ValueError(f'模型名字不能为空喵！')
    if copy_dict['model']['type'].casefold() not in SUPPORTED_MODEL_TYPE:
        raise ValueError(f"你真的确定 model.type: {copy_dict['model']['type']} 这个类型没有问题嘛，请检查相应的supported文件和目录喵~")
    # model.model_path字段
    if not copy_dict['model']['model_path'].is_dir():
        raise ValueError(f"该目录 model.model_path: {copy_dict['model']['model_path']} 不存在喵！请好好检查！")
    else:
        is_empty = not any(copy_dict['model']['model_path'].iterdir())
        if not is_empty:
            pass
        else:
            raise ValueError(f"该目录 model.model_path: {copy_dict['model']['model_path']} 下面不存在任何文件喵！")
    # seed 字段
    if not type(copy_dict['seed']) is int or copy_dict['seed'] <= 0:
        raise ValueError(f"种子必须要正整数喵！")
    # modules 字段
    # modules.order 字段
    if not isinstance(copy_dict['modules']['order'], list):
        raise ValueError(f"modules.order: {copy_dict['modules']['order']} 的值必须是列表类型的喵！")
    if len(copy_dict['modules']['order']) == 0:
        raise ValueError(f"modules.order: {copy_dict['modules']['order']} 的列表内不能为空喵！")
    if not all(isinstance(module, str) for module in copy_dict['modules']['order']):
        raise ValueError(f"modules.order: {copy_dict['modules']['order']} 里面的元素都必须要是字符串才行喵！")
    if not all(module.strip() for module in copy_dict['modules']['order']):
        raise ValueError(f"modules.order: {copy_dict['modules']['order']} 里面的元素都不能为空喵！")
    if not all(elem.casefold() in SUPPORTED_MODULES for elem in copy_dict['modules']['order']):
        raise ValueError(f"modules.order: {copy_dict['modules']['order']} 里面的元素不合法喵！找不到里面的模块喵！")
    # modules.layer_nums 字段
    if not isinstance(copy_dict['modules']['layer_nums'],list):
        raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 的值必须是列表类型的喵！")
    if len(copy_dict['modules']['layer_nums']) == 0:
        raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 里面的值不能为空喵！")
    if not all(type(module) is int for module in copy_dict['modules']['layer_nums']):
        raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 里面的元素都必须要是整型才行喵！")
    if not all(num > 0 for num in copy_dict['modules']['layer_nums']):
        raise ValueError(f"modules.layer_nums: {copy_dict['modules']['layer_nums']} 里面的元素都必须是正整数喵！")
    # modules.layer_nums和modules.order 比较字段
    if not len(copy_dict['modules']['order']) == len(copy_dict['modules']['layer_nums']):
        raise ValueError(f"modules.order: {copy_dict['modules']['order']} 和 modules.layer_nums: {copy_dict['modules']['layer_nums']} 的列表层数不一样喵！请好好检查这两个参数喵！")
    # task 字段
    if not isinstance(copy_dict['task'],str):
        raise ValueError(f"task: {copy_dict['task']} 的值必须是字符串类型的喵！")
    if not copy_dict['task'].strip():
        raise ValueError(f"task: {copy_dict['task']} 的值不能为空喵！")
    if not copy_dict['task'].casefold() in SUPPORTED_TASKS:
        raise ValueError(f"task: {copy_dict['task']} 的任务类型不受支持喵！")
    # cuda 字段
    # cuda.gpu_nums 字段
    if not type(copy_dict['cuda']['gpu_nums']) is int:
        raise ValueError(f"cuda.gpu_nums: {copy_dict['cuda']['gpu_nums']} 的值必须是整型的喵！")
    if copy_dict['cuda']['gpu_nums'] <= 0:
        raise ValueError(f"cuda.gpu_nums: {copy_dict['cuda']['gpu_nums']} 的值必须是正整数的喵！0也不行喵！")
    # cuda.num_workers 字段
    if not type(copy_dict['cuda']['num_workers'] is int):
        raise ValueError(f"cuda.num_workers: {copy_dict['cuda']['num_workers']} 的值必须是整型的喵！")
    if copy_dict['cuda']['num_workers'] < 0:
        raise ValueError(f"cuda.num_workers: {copy_dict['cuda']['num_workers']} 的值必须是正整数或者0才行的喵！")
    # train 字段
    # train.dataset 字段
    # train.dataset.type 字段
    if not isinstance(copy_dict['train']['dataset']['type'],str):
        raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的值必须是字符串类型的喵！")
    if not copy_dict['train']['dataset']['type'].strip():
        raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的值不能为空喵！")
    if not copy_dict['train']['dataset']['type'].casefold() in SUPPORTED_DATASET:
        raise ValueError(f"train.dataset.type: {copy_dict['train']['datset']['type']} 的数据集类型不受支持喵！")
