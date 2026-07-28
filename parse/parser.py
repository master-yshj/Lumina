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
    'dataset_path': 'REQUIRED',
    'train' : {
        'dataset': {
            'type': 'REQUIRED',
            'dataset_train_path': 'REQUIRED',
        },
        'batch_size': 8,
        'accumulation_steps': 1,
        'total_iters': 10000,
        'loss': {
            'CrossEntropyLoss': 1.0,
        },
        'optimizer': 'AdamW',
        'scheduler': 'StepLR',
        "embedding": {
            "type": "BertEmbedding",
            "hidden_size": 128,
            "max_position_embeddings": 128,
            "type_vocab_size": 2,
            "dropout": 0.1,
            "layer_norm_eps": 1.0e-12,
        },
        'lr': 1.0e-3,
    },
    'valid': {
            'dataset': {
                'type': 'REQUIRED',
                'dataset_val_path': 'REQUIRED',
            },
            'valid_iters': 500,
            'early_stop_patience': 10,
            'validation_rounds_before_save_ckpt': 1,
        } ,
    'test': {
        'dataset': {
            'type': 'REQUIRED',
            'dataset_test_path': 'REQUIRED',
        }
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
                                  SUPPORTED_TOKENIZERS,
                                  SUPPORTED_OPTIMIZERS,
                                  SUPPORTED_SCHEDULERS,
                                  SUPPORTED_EMBEDDINGS)

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
    path_list = ["model_path","dataset_path","dataset_train_path","dataset_val_path","dataset_test_path","vocab_file_path"]
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
    if not type(copy_dict['cuda']['num_workers']) is int:
        raise ValueError(f"cuda.num_workers: {copy_dict['cuda']['num_workers']} 的值必须是整型的喵！")
    if copy_dict['cuda']['num_workers'] < 0:
        raise ValueError(f"cuda.num_workers: {copy_dict['cuda']['num_workers']} 的值必须是正整数或者0才行的喵！")

    # dataset_path 字段
    if not copy_dict['dataset_path'].is_dir():
        raise ValueError(
            f"dataset_path {copy_dict['dataset_path']} 不是目录喵！好好检查喵！")
    else:
        is_empty = not any(copy_dict['dataset_path'].iterdir())
        if is_empty:
            raise ValueError(
                f"dataset_path: {copy_dict['dataset_path']} 该目录下面没有找到有效文件喵！")
        else:
            pass

    # train 字段
    # train.dataset 字段
    # train.dataset.type 字段
    if not isinstance(copy_dict['train']['dataset']['type'],str):
        raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的值必须是字符串类型的喵！")
    if not copy_dict['train']['dataset']['type'].strip():
        raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的值不能为空喵！")
    if not copy_dict['train']['dataset']['type'].casefold() in SUPPORTED_DATASET:
        raise ValueError(f"train.dataset.type: {copy_dict['train']['dataset']['type']} 的数据集类型不受支持喵！")
    # train.dataset.dataset_train_path 字段
    if not copy_dict['train']['dataset']['dataset_train_path'].is_file():
        raise ValueError(f"train.dataset.dataset_train_path: {copy_dict['train']['dataset']['dataset_train_path']} 指向的不是一个文件喵！好好检查喵！")
    if copy_dict['train']['dataset']['dataset_train_path'].stat().st_size == 0:
        raise ValueError(f"train.dataset.dataset_train_path {copy_dict['train']['dataset']['dataset_train_path']} 指向的文件是空的喵！")
    with open(copy_dict['train']['dataset']['dataset_train_path'], 'r', encoding='utf-8') as f:
        if not any(line.strip() for line in f):
            raise ValueError(f"train.dataset.dataset_train_path {copy_dict['train']['dataset']['dataset_train_path']} 指向的文件是空的喵！")
    # train.batch_size 字段
    if not type(copy_dict['train']['batch_size']) is int or copy_dict['train']['batch_size'] <= 0:
        raise ValueError(f"train.batch_size: {copy_dict['train']['batch_size']} 的值必须为正整数喵！")
    # train.accumulation_steps 字段
    if not type(copy_dict['train']['accumulation_steps']) is int or copy_dict['train']['accumulation_steps'] <= 0:
        raise ValueError(f"train.accumulation_steps: {copy_dict['train']['accumulation_steps']} 的值必须为正整数喵！")
    # train.total_iters 字段
    if not type(copy_dict['train']['total_iters']) is int or copy_dict['train']['total_iters'] <= 0:
        raise ValueError(f"train.total_iters: {copy_dict['train']['total_iters']} 的值必须为正整数喵！")
    # train.loss 字段
    if not isinstance(copy_dict['train']['loss'], dict):
        raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的值必须为字典喵！")
    if not copy_dict['train']['loss']:
        raise ValueError(f"train.loss: {copy_dict['train']['loss']} 里面是空的喵！")
    for key,value in copy_dict['train']['loss'].items():
        if not isinstance(key, str):
            raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的键必须为字符串类型的喵！")
        if not key.strip():
            raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的键不能为空喵！")
        if (type(value) is float or type(value) is int) and value > 0:
            pass
        else:
            raise ValueError(f"train.loss: {copy_dict['train']['loss']} 的值是损失的权重喵！必须为正数喵！")
        if not key.casefold() in SUPPORTED_LOSSES:
            raise ValueError(f"train.loss: {copy_dict['train']['loss']} 里面的 {key} 这种损失不受支持喵！请检查是否存在拼写错误（比如不能有下划线）")
    # train.optimizer 字段
    if not isinstance(copy_dict['train']['optimizer'], str):
        raise ValueError(f"train.optimizer: {copy_dict['train']['optimizer']} 的值必须为字符串类型的喵！")
    if not copy_dict['train']['optimizer'].strip():
        raise ValueError(f"train.optimizer: {copy_dict['train']['optimizer']} 的值不能为空喵！")
    if not copy_dict['train']['optimizer'].casefold() in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"train.optimizer: {copy_dict['train']['optimizer']} 选择的优化器不受支持喵！")
    # train.scheduler 字段
    if not isinstance(copy_dict['train']['scheduler'], str):
        raise ValueError(f"train.scheduler: {copy_dict['train']['scheduler']} 的值必须为字符串类型的喵！")
    if not copy_dict['train']['scheduler'].strip():
        raise ValueError(f"train.scheduler: {copy_dict['train']['scheduler']} 的值不能为空喵！")
    if not copy_dict['train']['scheduler'].casefold() in SUPPORTED_SCHEDULERS:
        raise ValueError(f"train.scheduler: {copy_dict['train']['scheduler']} 选择的调度器不受支持喵！")
    # train.embedding 字段
    # train.embedding.type 字段
    if not isinstance(copy_dict['train']['embedding']['type'], str):
        raise ValueError(f"train.embedding.type: {copy_dict['train']['embedding']['type']} 的值必须为字符串类型的喵！")
    if not copy_dict['train']['embedding']['type'].strip():
        raise ValueError(f"train.embedding.type: {copy_dict['train']['embedding']['type']} 的值不能为空喵！")
    if not copy_dict['train']['embedding']['type'].casefold() in SUPPORTED_EMBEDDINGS:
        raise ValueError(f"train.embedding.type: {copy_dict['train']['embedding']['type']} 选择的嵌入层不受支持喵！")
    # train.embedding.hidden_size 字段
    if not type(copy_dict['train']['embedding']['hidden_size']) is int or copy_dict['train']['embedding']['hidden_size'] <= 0:
        raise ValueError(f"train.embedding.hidden_size: {copy_dict['train']['embedding']['hidden_size']} 的值必须为正整数喵！")
    # train.embedding.max_position_embeddings 字段
    if not type(copy_dict['train']['embedding']['max_position_embeddings']) is int or copy_dict['train']['embedding']['max_position_embeddings'] <= 0:
        raise ValueError(f"train.embedding.max_position_embeddings: {copy_dict['train']['embedding']['max_position_embeddings']} 的值必须为正整数喵！")
    # train.embedding.type_vocab_size 字段
    if not type(copy_dict['train']['embedding']['type_vocab_size']) is int or copy_dict['train']['embedding']['type_vocab_size'] <= 0:
        raise ValueError(f"train.embedding.type_vocab_size: {copy_dict['train']['embedding']['type_vocab_size']} 的值必须为正整数喵！")
    # train.embedding.dropout 字段
    if not type(copy_dict['train']['embedding']['dropout']) in (int, float) or not 0 <= copy_dict['train']['embedding']['dropout'] < 1:
        raise ValueError(f"train.embedding.dropout: {copy_dict['train']['embedding']['dropout']} 的值必须为 [0,1) 之间的 int 或 float 喵！")
    # train.embedding.layer_norm_eps 字段
    if not type(copy_dict['train']['embedding']['layer_norm_eps']) in (int, float) or copy_dict['train']['embedding']['layer_norm_eps'] <= 0:
        raise ValueError(f"train.embedding.layer_norm_eps: {copy_dict['train']['embedding']['layer_norm_eps']} 的值必须为大于0的 int 或 float 喵！")
    # train.lr 字段
    if type(copy_dict['train']['lr']) not in (int,float) or copy_dict['train']['lr'] <= 0:
        raise ValueError(f"train.lr: {copy_dict['train']['lr']} 的值必须为正数喵！")

    # valid 字段
    # valid.dataset 字段
    # valid.dataset.type 字段
    if not isinstance(copy_dict['valid']['dataset']['type'],str):
        raise ValueError(f"valid.dataset.type: {copy_dict['valid']['dataset']['type']} 的值必须为字符串类型的喵！")
    if not copy_dict['valid']['dataset']['type'].strip():
        raise ValueError(f"valid.dataset.type: {copy_dict['valid']['dataset']['type']} 的值不能为空喵！")
    if not copy_dict['valid']['dataset']['type'].casefold() in SUPPORTED_DATASET:
        raise ValueError(f"valid.dataset.type: {copy_dict['valid']['dataset']['type']} 这个数据集不受支持喵！")
    # valid.dataset.dataset_val_path 字段
    if not copy_dict['valid']['dataset']['dataset_val_path'].is_file():
        raise ValueError(f"valid.dataset.dataset_val_path: {copy_dict['valid']['dataset']['dataset_val_path']} 指向的不是一个文件喵！好好检查喵！")
    if copy_dict['valid']['dataset']['dataset_val_path'].stat().st_size == 0:
        raise ValueError(f"valid.dataset.dataset_val_path {copy_dict['valid']['dataset']['dataset_val_path']} 指向的文件是空的喵！")
    with open(copy_dict['valid']['dataset']['dataset_val_path'], 'r', encoding='utf-8') as f:
        if not any(line.strip() for line in f):
            raise ValueError(f"valid.dataset.dataset_val_path {copy_dict['valid']['dataset']['dataset_val_path']} 指向的文件是空的喵！")
    # valid.valid_iters 字段
    if not type(copy_dict['valid']['valid_iters']) is int or copy_dict['valid']['valid_iters'] <= 0:
        raise ValueError(f"valid.valid_iters: {copy_dict['valid']['valid_iters']} 的值必须要为正整数喵！")
    # valid.early_stop_patience
    if not type(copy_dict['valid']['early_stop_patience']) is int or copy_dict['valid']['early_stop_patience'] <= 0:
        raise ValueError(f"valid.early_stop_patience: {copy_dict['valid']['early_stop_patience']} 的值必须要为正整数喵！")
    # valid.validation_rounds_before_save_ckpt 字段
    if not type(copy_dict['valid']['validation_rounds_before_save_ckpt']) is int or copy_dict['valid']['validation_rounds_before_save_ckpt'] <= 0:
        raise ValueError(f"valid.validation_rounds_before_save_ckpt: {copy_dict['valid']['validation_rounds_before_save_ckpt']} 的值必须为正整数喵！")

    # test 字段
    # test.dataset 字段
    # test.dataset.type 字段
    if not isinstance(copy_dict['test']['dataset']['type'],str):
        raise ValueError(f"test.dataset.type: {copy_dict['test']['dataset']['type']} 的值必须为字符串类型的喵！")
    if not copy_dict['test']['dataset']['type'].strip():
        raise ValueError(f"test.dataset.type: {copy_dict['test']['dataset']['type']} 的值不能为空喵！")
    if not copy_dict['test']['dataset']['type'].casefold() in SUPPORTED_DATASET:
        raise ValueError(f"test.dataset.type: {copy_dict['test']['dataset']['type']} 这个数据集不受支持喵！")
    # test.dataset.dataset_test_path 字段
    if not copy_dict['test']['dataset']['dataset_test_path'].is_file():
        raise ValueError(f"test.dataset.dataset_test_path: {copy_dict['test']['dataset']['dataset_test_path']} 指向的不是一个文件喵！好好检查喵！")
    if copy_dict['test']['dataset']['dataset_test_path'].stat().st_size == 0:
        raise ValueError(f"test.dataset.dataset_test_path {copy_dict['test']['dataset']['dataset_test_path']} 指向的文件是空的喵！")
    with open(copy_dict['test']['dataset']['dataset_test_path'], 'r', encoding='utf-8') as f:
        if not any(line.strip() for line in f):
            raise ValueError(f"test.dataset.dataset_test_path {copy_dict['test']['dataset']['dataset_test_path']} 指向的文件是空的喵！")

    # tokenizer 字段
    # tokenizer.type 字段
    if not isinstance(copy_dict['tokenizer']['type'], str):
        raise ValueError(f"tokenizer.type: {copy_dict['tokenizer']['type']} 的值必须为字符串类型的喵！")
    if not copy_dict['tokenizer']['type'].strip():
        raise ValueError(f"tokenizer.type: {copy_dict['tokenizer']['type']} 的值不能为空喵！")
    if not copy_dict['tokenizer']['type'].casefold() in SUPPORTED_TOKENIZERS:
        raise ValueError(f"tokenizer.type: {copy_dict['tokenizer']['type']} 这种tokenizer不受支持喵！")
    # tokenizer.vocab_file_path 字段
    if not copy_dict['tokenizer']['vocab_file_path'].is_file():
        raise ValueError(f"tokenizer.vocab_file_path: {copy_dict['tokenizer']['vocab_file_path']} 指向的不是一个文件喵！请好好检查喵！")
    if copy_dict['tokenizer']['vocab_file_path'].stat().st_size == 0:
        raise ValueError(f"tokenizer.vocab_file_path: {copy_dict['tokenizer']['vocab_file_path']} 指向的文件是空的喵！")
    with open(copy_dict['tokenizer']['vocab_file_path'], 'r', encoding='utf-8') as f:
        if not any(line.strip() for line in f):
            raise ValueError(f"tokenizer.vocab_file_path: {copy_dict['tokenizer']['vocab_file_path']} 指向的文件是空的喵！")