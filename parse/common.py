from pathlib import Path
from .errors import ConfigError

FieldPath = tuple[str | int, ...]


def config_error(message: str, field_path: FieldPath = ()) -> ConfigError:
    """
    :param message: str, 配置错误的具体说明
    :param field_path: FieldPath = tuple[str | int, ...], 出错字段在配置树中的位置，默认为 ()，表示根节点

    :return: ConfigError, 保存错误信息和 field_path 属性的异常对象

    该函数用于构造带字段位置的配置异常，返回给调用方抛出，本函数不执行抛出操作
    """
    error = ConfigError(message)
    error.field_path = field_path
    return error


def nearest_origin(origins: dict, field_path: FieldPath) -> dict:
    """
    :param origins: dict, 键为字段位置元组，值为记录 kind、source_path、base_dir 等信息的来源字典
    :param field_path: FieldPath, 要查询来源的字段位置，例如 ('run', 'mode')

    :return: dict, 当前字段或最近祖先节点的来源记录，找不到时返回 {}，找到时返回原记录而非副本

    该函数用于先查询当前字段的来源，未记录时逐级查父节点，供错误报告和相对路径解析使用
    """
    while True:
        if field_path in origins:
            return origins[field_path]
        if not field_path:
            return {}
        field_path = field_path[:-1]


def format_field_path(field_path: FieldPath) -> str:
    """
    :param field_path: FieldPath, 配置字段位置元组，字符串表示字典键，整数表示列表或元组下标

    :return: str, 用点号连接字段、用方括号表示下标的位置文字，空元组返回 <root>

    该函数用于把字段位置转换为可读文字，例如 ('losses', 0, 'weight') 转成 losses[0].weight
    """
    text = ''
    for part in field_path:
        text += f'[{part}]' if isinstance(part, int) else ('.' if text else '') + part
    return text or '<root>'


def source_description(origin: dict) -> str:
    """
    :param origin: dict, 单个字段的来源记录，可含 kind、source_path、base_dir、schema 和 transformation

    :return: str, YAML 文件、CLI 或 Schema 默认值的来源说明，未识别来源时返回“来源未记录”

    该函数用于将来源记录转换为报错用的文字，并在 transformation 非空时追加解析器对该字段的处理说明
    """
    kind = origin.get('kind')
    if kind == 'schema_default':
        text = f"Schema 默认 {origin.get('schema')}，基准目录 {origin.get('base_dir')}"
    elif kind == 'cli':
        text = f"CLI，基准目录 {origin.get('base_dir')}"
    elif kind == 'yaml':
        text = f"YAML {origin.get('source_path')}"
    else:
        text = '来源未记录'
    if origin.get('transformation'):
        text += f"；{origin['transformation']}"
    return text


def origin_base(origins: dict, field_path: FieldPath, fallback: Path) -> Path:
    """
    :param origins: dict, 键为字段位置元组、值为来源信息字典的来源表
    :param field_path: FieldPath, 要查找路径基准的字段位置，不是文件系统路径
    :param fallback: Path, 来源记录缺少 base_dir 键时使用的备用绝对目录

    :return: Path, 来源记录中的绝对基准目录，或备用绝对目录

    该函数用于先查来源再取 base_dir，确认它是绝对 Path，供后续拼接相对文件路径使用
    """
    base = nearest_origin(origins, field_path).get('base_dir', fallback)
    if not isinstance(base, Path) or not base.is_absolute():
        raise config_error('来源基准必须为绝对 Path，请人工检查 origins 喵!', field_path)
    return base
