from copy import deepcopy
from pathlib import Path
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from .common import config_error, nearest_origin, origin_base
from .errors import ConfigError
from .schema_registry import resolve_task_key


def _unwrap(annotation):
    """
    :param annotation: object, 从配置类的字段声明中取得的类型注解，例如 Annotated[float, Field(...)]

    :return: tuple[object, list], 按顺序返回去掉外层 Annotated 的类型注解和其中附带的元数据列表

    该函数用于拆开类型注解和附加规则，供后续选择配置类及读取规范化标记使用，不处理配置值
    """
    metadata = []
    while get_origin(annotation) is Annotated:
        annotation, *extra = get_args(annotation)
        metadata.extend(extra)
    return annotation, metadata


def _is_model(annotation) -> bool:
    """
    :param annotation: object, 待判断的类型注解

    :return: bool, 是 BaseModel 的子类对象时为 True，否则为 False

    该函数用于区分配置类、普通类型与配置实例，使遍历器只对配置类读取 model_fields
    """
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _canonical_name(value, category, field_path, catalog):
    """
    :param value: str, 配置中填写的组件名称或别名
    :param category: str | None, 运行注册表的类别键，例如 tokenizer；为空时只处理字符串
    :param field_path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径
    :param catalog: dict, build_schema_catalog 返回的规则表，包含配置类、组件注册表和 Union 分支选择规则

    :return: str, 去除首尾空白并 casefold 后的名称；指定类别时返回注册表中的规范名

    该函数用于通过运行注册表解析名称及别名，失败时把字段位置附在错误上，不修改传入节点
    """
    if not isinstance(value, str) or not value.strip():
        raise config_error('选择组件的名称必须为非空字符串。', field_path)
    name = value.strip().casefold()
    if category:
        try:
            name = catalog['runtime_registries'][category].resolve_key(name)
        except ConfigError as error:
            error.field_path = field_path
            raise
    return name


def select_annotation(annotation, node, field_path, catalog) -> tuple:
    """
    :param annotation: object, 当前字段声明的类型注解，可能包含 Annotated、Union 或配置类
    :param node: object, 当前字段的实际配置值，用于选择适用的类型分支
    :param field_path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径
    :param catalog: dict, build_schema_catalog 返回的规则表，包含配置类、组件注册表和 Union 分支选择规则

    :return: tuple[object, list], 按顺序返回选中的类型注解和提取出的 Annotated 元数据

    该函数用于按空值、组件名称或数据形状选择遍历规则，并检查组件是否有对应参数规则；不构造配置实例，也不执行完整的值范围校验
    """
    annotation, metadata = _unwrap(annotation)
    if get_origin(annotation) in (Union, UnionType):
        args = get_args(annotation)
        if type(None) in args:
            if node is None:
                return type(None), metadata
            remaining = tuple(arg for arg in args if arg is not type(None))
            reduced = remaining[0] if len(remaining) == 1 else Union[remaining]
            chosen, extra = select_annotation(reduced, node, field_path, catalog)
            return chosen, metadata + extra
        route = catalog['union_routes'].get(annotation)
        if route:
            if not isinstance(node, dict):
                raise config_error('动态组件必须为 dict，才能依据 type 选择 params 规则。', field_path)
            discriminator = route['discriminator']
            exemplar = next(iter(route['variants'].values()))
            hints = exemplar.model_fields[discriminator].json_schema_extra or {}
            name = _canonical_name(node.get(discriminator), hints.get('registry'), field_path + (discriminator,), catalog)
            if name not in route['variants']:
                raise config_error(f'组件 {name!r} 已有运行登记，但当前上下文缺少参数规则。', field_path + (discriminator,))
            return route['variants'][name], metadata
        for arg in args:
            plain, extra = _unwrap(arg)
            origin = get_origin(plain)
            matches = (
                (plain is int and type(node) is int) or (plain is float and type(node) in (int, float))
                or (plain is str and isinstance(node, str)) or (plain is Path and isinstance(node, (str, Path)))
                or (origin is Literal and any(type(v) is type(node) for v in get_args(plain)))
                or (origin is list and isinstance(node, list)) or (origin is tuple and isinstance(node, (list, tuple)))
            )
            if matches:
                return plain, metadata + extra
        if any(_is_model(_unwrap(arg)[0]) for arg in args):
            raise config_error('准备器不支持没有 discriminator 路由的模型联合，请检查登记。', field_path)
        return annotation, metadata
    if _is_model(annotation) and isinstance(node, dict):
        for name, field in annotation.model_fields.items():
            hints = field.json_schema_extra or {}
            if hints.get('registry') and get_origin(field.annotation) is Literal:
                canonical = _canonical_name(node.get(name), hints['registry'], field_path + (name,), catalog)
                if canonical not in get_args(field.annotation):
                    raise config_error(f'组件 {canonical!r} 在此上下文没有参数规则；允许 {get_args(field.annotation)}。', field_path + (name,))
    return annotation, metadata


def _check_plain(value, field_path):
    """
    :param value: object, 待检查的配置节点或默认值
    :param field_path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径

    :return: None, 检查成功直接结束，不修改输入

    该函数用于递归允许字符串键的 dict、list、tuple、None、str、int、float、bool 和 Path，防止准备阶段提前混入 BaseModel 实例或其他对象
    """
    if isinstance(value, BaseModel):
        raise config_error('准备树或默认工厂中出现 BaseModel 实例，配置实例只能在最终校验创建。', field_path)
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise config_error('配置 dict 的键必须是字符串。', field_path)
            _check_plain(child, field_path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_plain(child, field_path + (index,))
    elif value is not None and not isinstance(value, (str, int, float, bool, Path)):
        raise config_error(f'准备树包含不支持的类型 {type(value).__name__}。', field_path)


def get_plain_default(field: FieldInfo, field_path) -> object:
    """
    :param field: FieldInfo, 从配置类 model_fields 中取出的字段定义，必须带默认值或默认工厂
    :param field_path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径

    :return: object, 默认值的深拷贝，可以是普通容器、标量、Path 或 None

    该函数用于调用 Pydantic 的字段接口取得默认值并检查其类型，供准备层填充缺失字段，不创建子配置实例
    """
    if field.is_required():
        raise config_error('必填字段没有默认值，请检查准备器调用。', field_path)
    if field.default_factory_takes_validated_data:
        raise config_error('默认工厂不能依赖 validated_data，准备阶段尚未进行校验。', field_path)
    value = deepcopy(field.get_default(call_default_factory=True))
    _check_plain(value, field_path)
    return value


def _element_annotation(annotation, index):
    """
    :param annotation: object, list 或 tuple 的类型注解，例如 list[int]、tuple[float, float]
    :param index: int, 当前元素从 0 开始的位置

    :return: object | None, 该位置对应的元素类型注解；没有对应注解时返回 None

    该函数用于为列表、定长元组和 tuple[T, ...] 选择元素规则，保留超出定长元组长度的元素供最终校验发现
    """
    args = get_args(annotation)
    if get_origin(annotation) is list:
        return args[0] if args else None
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    return args[index] if index < len(args) else None


def fill_defaults(raw_config, root_schema, origins, catalog, default_base_dir) -> tuple[dict, dict]:
    """
    :param raw_config: dict, 合并和插值处理后、尚未填默认值的普通配置树
    :param root_schema: type[BaseModel], 当前任务及运行模式对应的根配置类
    :param origins: dict, 键为字段位置元组，值为该字段的来源信息字典
    :param catalog: dict, build_schema_catalog 返回的规则表，包含配置类、组件注册表和 Union 分支选择规则
    :param default_base_dir: Path, 主 YAML 所在目录的绝对路径，字段来源没有基准目录时使用

    :return: tuple[dict, dict], 按顺序返回补齐默认值的配置树和包含默认字段来源的新来源表

    该函数用于按配置类递归填充不存在且非必填的字段，显式 None 和未知键保留；深拷贝输入，不修改调用方的配置与来源表
    """
    _check_plain(raw_config, ())
    filled_origins = deepcopy(origins)

    def record_default(value, path, info):
        """
        :param value: object, 刚取得的默认值或其子节点
        :param path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径
        :param info: dict, 默认值的来源记录，包含 kind、source_path、base_dir 和声明默认值的 schema

        :return: None, 结果写入外层函数的 filled_origins 字典

        该函数用于给默认值及其所有子节点记录来源，每个节点保存独立的信息字典，便于错误定位和相对路径解析
        """
        filled_origins[path] = info.copy()
        children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
        for key, child in children:
            record_default(child, path + (key,), info)

    def walk(node, annotation, path):
        """
        :param node: object, 深拷贝后的配置节点
        :param annotation: object, 当前节点对应的类型注解或配置类
        :param path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径

        :return: object, 当前节点补齐默认值后的结果，容器保留原来的 dict、list 或 tuple 形式

        该函数为fill_defaults的内部函数, 用于读取外层 catalog 选择规则，遍历已知字段并填入缺失默认值，借助 record_default 更新 filled_origins，默认路径基准来自 default_base_dir 或现有来源
        """
        selected, _ = select_annotation(annotation, node, path, catalog)
        if node is None:
            return None
        if _is_model(selected):
            if not isinstance(node, dict):
                raise config_error('该配置节点必须为普通 dict。', path)
            result = node.copy()
            for name, field in selected.model_fields.items():
                child_path = path + (name,)
                if name not in result and not field.is_required():
                    result[name] = get_plain_default(field, child_path)
                    record_default(result[name], child_path, {
                        'kind': 'schema_default', 'source_path': None,
                        'base_dir': origin_base(filled_origins, path, default_base_dir),
                        'schema': f'{selected.__name__}.{name}',
                    })
                if name in result:
                    result[name] = walk(result[name], field.annotation, child_path)
            return result
        if get_origin(selected) in (list, tuple) and isinstance(node, (list, tuple)):
            result = []
            for index, child in enumerate(node):
                element = _element_annotation(selected, index)
                result.append(walk(child, element, path + (index,)) if element is not None else child)
            return tuple(result) if isinstance(node, tuple) else result
        return node

    return walk(deepcopy(raw_config), root_schema, ()), filled_origins


def normalize_path(value, field_path, origins, default_base_dir, dataset_root=None):
    """
    :param value: str | Path | None, 待规范化的路径字段值，None 原样保留
    :param field_path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径
    :param origins: dict, 键为字段位置元组，值为该字段的来源信息字典
    :param default_base_dir: Path, 主 YAML 所在目录的绝对路径，字段来源没有基准目录时使用
    :param dataset_root: Path | None, 数据划分使用的数据集根目录绝对路径，默认 None 表示按字段来源确定基准

    :return: Path | None, 规范化后的绝对路径；输入 None 时返回 None

    该函数用于去除路径字符串首尾空白、转为 Path 并按基准目录绝对化，此处不检查目标是否存在或是否为文件
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise config_error('路径字符串不能为空。', field_path)
        value = Path(value.strip())
    elif not isinstance(value, Path):
        raise config_error(f'路径必须为 str 或 Path，实际为 {type(value).__name__}。', field_path)
    base = dataset_root if dataset_root is not None else origin_base(origins, field_path, default_base_dir)
    if not isinstance(base, Path) or not base.is_absolute():
        raise config_error('路径基准必须已经是绝对 Path。', field_path)
    try:
        return (value if value.is_absolute() else base / value).resolve(strict=False)
    except (OSError, ValueError, RuntimeError) as error:
        raise config_error(f'路径无法绝对化：{error}', field_path) from error


def _hints(field_info, metadata) -> dict:
    """
    :param field_info: FieldInfo | None, 当前字段的 Pydantic 定义；遍历容器元素时可能为 None
    :param metadata: list, 从 Annotated 中取出的附加规则，其中可能包含 FieldInfo

    :return: dict, 合并后的项目标记，例如 normalize、path_base 和 resource

    该函数用于读取元数据及字段定义中的 json_schema_extra，后读到的同名键覆盖前者，字段本身的标记优先
    """
    hints = {}
    for item in [*metadata, field_info]:
        if isinstance(item, FieldInfo) and isinstance(item.json_schema_extra, dict):
            hints.update(item.json_schema_extra)
    return hints


def normalize_tree(filled_config, root_schema, filled_origins, catalog, default_base_dir) -> dict:
    """
    :param filled_config: dict, 已完成默认值填充的普通配置树
    :param root_schema: type[BaseModel], 当前任务及模式对应的根配置类
    :param filled_origins: dict, 键为字段位置元组，值为该字段的来源信息字典
    :param catalog: dict, build_schema_catalog 返回的规则表，包含配置类、组件注册表和 Union 分支选择规则
    :param default_base_dir: Path, 主 YAML 所在目录的绝对路径，字段来源没有基准目录时使用

    :return: dict, 名称、路径及指定表示形式规范化后的新配置树

    该函数用于深拷贝后按字段标记处理名称、路径、元组和浮点数；先处理 dataset.root 再处理划分路径，保留未知键供最终校验
    """
    def walk(node, annotation, field_info, path, dataset_root):
        """
        :param node: object, 当前需要规范化的配置节点
        :param annotation: object, 当前节点的类型注解或配置类
        :param field_info: FieldInfo | None, 当前字段定义；根节点和容器元素没有独立字段定义时为 None
        :param path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径
        :param dataset_root: Path | None, 已规范化的数据集根目录，供 path_base 为 dataset_root 的字段使用

        :return: object, 当前节点及其已知子字段规范化后的结果

        该函数为normalize_tree的内部函数, 用于读取外层 catalog、filled_origins 和 default_base_dir，按标记转换当前值并递归返回新容器；处理数据集时先取得 root 的绝对路径再传给 splits
        """
        selected, metadata = select_annotation(annotation, node, path, catalog)
        hints = _hints(field_info, metadata)
        action = hints.get('normalize')
        if node is None:
            return None
        if action in ('strip', 'name') and isinstance(node, str):
            node = node.strip()
            if action == 'name':
                node = node.casefold()
                if path == ('task', 'type'):
                    node = resolve_task_key(node, catalog['task_specs'], path)
                elif hints.get('registry'):
                    node = _canonical_name(node, hints['registry'], path, catalog)
        elif action == 'path':
            if hints.get('path_base') == 'dataset_root' and not isinstance(dataset_root, Path):
                raise config_error('规范化 split 前必须先得到 dataset.root 的绝对 Path。', path)
            node = normalize_path(node, path, filled_origins, default_base_dir,
                                  dataset_root if hints.get('path_base') == 'dataset_root' else None)
        elif action == 'tuple' and isinstance(node, (list, tuple)):
            node = tuple(node)
        if (action == 'float' or selected is float) and type(node) in (int, float):
            try:
                node = float(node)
            except OverflowError as error:
                raise config_error('数值过大，不能规范为 float。', path) from error
        if _is_model(selected):
            if not isinstance(node, dict):
                raise config_error('该配置节点必须为普通 dict。', path)
            result = node.copy()
            fields = list(selected.model_fields.items())
            if path == ('data', 'dataset'):
                fields.sort(key=lambda item: item[0] != 'root')
            for name, field in fields:
                if name in result:
                    result[name] = walk(result[name], field.annotation, field, path + (name,), dataset_root)
                    if path == ('data', 'dataset') and name == 'root':
                        dataset_root = result[name]
            return result
        if get_origin(selected) in (list, tuple) and isinstance(node, (list, tuple)):
            values = []
            for index, child in enumerate(node):
                element = _element_annotation(selected, index)
                values.append(walk(child, element, None, path + (index,), dataset_root) if element is not None else child)
            return tuple(values) if isinstance(node, tuple) else values
        return node

    return walk(deepcopy(filled_config), root_schema, None, (), None)


def check_prepared_tree(prepared_config, root_schema, catalog) -> None:
    """
    :param prepared_config: dict, 已完成默认值填充和规范化的配置树
    :param root_schema: type[BaseModel], 当前配置应遵循的根配置类
    :param catalog: dict, build_schema_catalog 返回的规则表，包含配置类、组件注册表和 Union 分支选择规则

    :return: None, 检查成功直接结束，不修改配置

    该函数用于检查默认字段是否补齐以及名称、绝对 Path、tuple 和 float 转换是否完成，避免最终 Pydantic 校验掩盖准备阶段的遗漏
    """
    _check_plain(prepared_config, ())

    def walk(node, annotation, field_info, path):
        """
        :param node: object, 当前待检查的配置节点
        :param annotation: object, 当前节点的类型注解或配置类
        :param field_info: FieldInfo | None, 当前字段定义，根节点及容器元素可为 None
        :param path: tuple[str | int, ...], 当前字段在配置树中的位置，例如 ("components", "losses", 0)，不是磁盘路径

        :return: None, 当前节点及其已知子节点通过检查后结束

        该函数为check_prepared_tree的内部函数, 用于读取外层 catalog 和字段标记，递归核对准备结果；只检查应完成的默认值及表示转换，字段值是否合法由后续校验负责
        """
        selected, metadata = select_annotation(annotation, node, path, catalog)
        hints = _hints(field_info, metadata)
        action = hints.get('normalize')
        if node is None:
            return
        if action in ('strip', 'name') and isinstance(node, str):
            expected = node.strip()
            if action == 'name':
                expected = expected.casefold()
                if path == ('task', 'type'):
                    expected = resolve_task_key(expected, catalog['task_specs'], path)
                elif hints.get('registry'):
                    expected = _canonical_name(expected, hints['registry'], path, catalog)
            if node != expected:
                raise config_error('准备层遗漏名称规范化。', path)
        if action == 'path' and (not isinstance(node, Path) or not node.is_absolute()):
            raise config_error('准备层应输出绝对 Path。', path)
        if (action == 'tuple' or get_origin(selected) is tuple) and isinstance(node, list):
            raise config_error('准备层遗漏 list→tuple 转换。', path)
        if (action == 'float' or selected is float) and type(node) is int:
            raise config_error('准备层遗漏 int→float 转换。', path)
        if _is_model(selected):
            if not isinstance(node, dict):
                raise config_error('准备树中的配置块必须为 dict。', path)
            for name, field in selected.model_fields.items():
                if name not in node:
                    if not field.is_required():
                        raise config_error('准备层漏填默认字段，不能交给 Pydantic 隐式补齐。', path + (name,))
                else:
                    walk(node[name], field.annotation, field, path + (name,))
        elif get_origin(selected) in (list, tuple) and isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                element = _element_annotation(selected, index)
                if element is not None:
                    walk(child, element, None, path + (index,))

    walk(prepared_config, root_schema, None, ())


def prepare_config(raw_config, root_schema, origins, catalog, default_base_dir) -> tuple[dict, dict]:
    """
    :param raw_config: dict, 合并和插值处理后的普通配置树
    :param root_schema: type[BaseModel], 当前任务及模式对应的根配置类
    :param origins: dict, 键为字段位置元组，值为该字段的来源信息字典
    :param catalog: dict, build_schema_catalog 返回的规则表，包含配置类、组件注册表和 Union 分支选择规则
    :param default_base_dir: Path, 主 YAML 所在目录的绝对路径，字段来源没有基准目录时使用

    :return: tuple[dict, dict], 按顺序返回规范化配置和包含默认字段记录的来源表

    该函数用于依次完成整树默认值填充、整树规范化及准备完整性检查，将结果交给最终校验；规范化或完整性检查失败时在错误上附带本次来源表
    """
    filled, filled_origins = fill_defaults(raw_config, root_schema, origins, catalog, default_base_dir)
    try:
        normalized = normalize_tree(filled, root_schema, filled_origins, catalog, default_base_dir)
        check_prepared_tree(normalized, root_schema, catalog)
    except ConfigError as error:
        error.config_origins = filled_origins
        raise
    return normalized, filled_origins
