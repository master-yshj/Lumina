from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from .errors import ConfigError
from yaml import YAMLError
from copy import deepcopy
from omegaconf.errors import OmegaConfBaseException

def resolve_config_path(config_path: str | Path) -> Path:
    """
    :param config_path: str | Path, 待读取的 YAML 文件路径，可来自主入口参数或 run.checkpoint.train_config

    :return: Path, 已确认存在且为普通 .yaml 或 .yml 文件的绝对路径

    该函数用于去除路径字符串首尾空白并绝对化，检查存在性、文件类型和后缀，供 YAML 加载使用
    """
    tmp_path = config_path
    if isinstance(config_path, str):
        normalized_path = config_path.strip()
        if not normalized_path:
            raise ValueError(f"config_path 为str类型时不能为空喵!")
        tmp_path = Path(normalized_path)
    elif isinstance(config_path, Path):
        pass
    else:
        raise TypeError(f"config_path 必须为pathlib.Path类型或str类型的喵! 现在的type: {type(config_path)}")
    resolved_path = tmp_path.resolve()
    if not resolved_path.suffix.casefold() == ".yaml" and not resolved_path.suffix.casefold() == ".yml":
        raise ValueError(f"resolved_path: {resolved_path} 必须指向一个yaml文件或yml文件喵!")
    if not resolved_path.exists():
        raise ValueError(f"resolved_path: {resolved_path} 该路径不存在喵!")
    if not resolved_path.is_file():
        raise ValueError(f"resolved_path: {resolved_path} 必须指向一个文件喵!")
    return resolved_path

def load_yaml(yaml_path: Path) -> DictConfig:
    """
    :param yaml_path: Path, 待读取 YAML 文件的路径，通常已经由 resolve_config_path 检查并绝对化

    :return: DictConfig, OmegaConf 读取出的配置映射，尚未统一解析插值或规范化业务字段

    该函数用于以 UTF-8 读取单份 YAML 并要求根节点为映射，供后续收集来源与合并使用
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            file_config = OmegaConf.load(f)
    except (YAMLError, OSError, UnicodeDecodeError, OmegaConfBaseException) as e:
        raise ConfigError(f"yaml读取失败了喵! yaml_path: {yaml_path} 具体报错信息: {e}")
    if not isinstance(file_config, DictConfig):
        raise ConfigError(f"yaml_path: {yaml_path} 该yaml被读取后应是 DictConfig 类型的喵! 现在的type: {type(file_config)}")
    return file_config

def parse_cli_overrides(cli_overrides: list[str]) -> DictConfig:
    """
    :param cli_overrides: list[str], 命令行覆盖表达式列表，例如 ['runtime.precision=bf16']，允许空列表

    :return: DictConfig, OmegaConf.from_dotlist 根据点分路径构造的嵌套配置，空输入得到空配置

    该函数用于检查赋值格式、重复键和父子路径冲突后解析 CLI 值，供后续覆盖 YAML，不在此检查业务数值范围
    """
    if not isinstance(cli_overrides, list):
        raise ConfigError(f"cli_overrides: {cli_overrides} 必须为list类型的喵! 现在的type: {type(cli_overrides)}")
    seen_paths: dict[tuple[str, ...], str] = {}
    for override in cli_overrides:
        if not isinstance(override, str):
            raise ConfigError(f"override: {override} 必须为str类型的喵! 现在的type: {type(override)}")
        if not "=" in override:
            raise ConfigError(f"override: {override} 中必须包含等号喵!")
        pre_override, post_override = override.split("=", maxsplit=1)
        if not pre_override.strip():
            raise ConfigError(f"override: {override} 等号左边的部分不能出现空白喵!")
        for single_str in pre_override:
            if single_str.isspace():
                raise ConfigError(f"override: {override} 等号左边的部分不能出现空白喵!")
        if any(char in pre_override for char in ("[", "]", chr(92))):
            raise ConfigError("CLI 仅支持点分键，不支持方括号或转义键；列表请整项赋值喵~")
        override_keys = pre_override.split(".")
        for override_key in override_keys:
            if not override_key.strip():
                raise ConfigError(f"override: {override} 等号左边的部分每个层级不能为空喵!")
            for single_str in override_key:
                if single_str.isspace():
                    raise ConfigError(f"override: {override} 等号左边的部分每个层级不能出现空白的部分喵!")
        current_path = tuple(override_keys)
        if current_path in seen_paths:
            raise ConfigError(f"CLI 字段重复赋值了喵! 前一项: {seen_paths[current_path]!r}，当前项: {override!r}")
        for previous_path, previous_override in seen_paths.items():
            if current_path[:len(previous_path)] == previous_path or previous_path[:len(current_path)] == current_path:
                raise ConfigError(f"CLI 字段存在父子路径冲突了喵! 前一项: {previous_override!r}，当前项: {override!r}")
        seen_paths[current_path] = override
    try:
        parsed_config = OmegaConf.from_dotlist(cli_overrides)
    except (YAMLError, OmegaConfBaseException) as e:
        raise ConfigError(f"cli_overrides: {cli_overrides} 的解析出现异常了喵! {e}")
    return parsed_config

def collect_origins(
    config: DictConfig | ListConfig,
    *,
    kind: str,
    source_path: Path | None,
    base_dir: Path,
    field_path: tuple[str | int, ...] = (),
) -> dict[tuple[str | int, ...], dict[str, str | Path | None]]:
    """
    :param config: DictConfig | ListConfig, 单份已经读取的 YAML 配置或已经解析的 CLI 配置
    :param kind: str, 本份配置的来源类别，只允许 yaml 或 cli
    :param source_path: Path | None, 来源 YAML 的绝对路径，kind 为 cli 时必须为 None
    :param base_dir: Path, 相对路径的绝对基准目录，YAML 使用其父目录，CLI 使用调用方指定的目录
    :param field_path: tuple[str | int, ...], 遍历起点在配置树中的位置，默认为 ()，表示根节点

    :return: dict, origins 来源表，键为节点位置元组，值为包含 kind、source_path、base_dir 的独立字典

    该函数用于为本份配置的根节点、容器及各子节点逐项记录来源，保留插值和 ???，不把来源信息写入配置值
    """
    if not isinstance(config, (DictConfig, ListConfig)):
        raise ConfigError(f"config 必须为 DictConfig 或 ListConfig 喵! 现在的type: {type(config)}")
    if kind not in ("yaml", "cli"):
        raise ConfigError(f"kind 必须为 yaml 或 cli 喵! 现在的值: {kind!r}")
    if not isinstance(base_dir, Path) or not base_dir.is_absolute():
        raise ConfigError(f"base_dir 必须为绝对 Path 喵! 现在的值: {base_dir!r}")
    if kind == "yaml":
        if not isinstance(source_path, Path) or not source_path.is_absolute():
            raise ConfigError(f"YAML 的 source_path 必须为绝对 Path 喵! 现在的值: {source_path!r}")
        if source_path.parent != base_dir:
            raise ConfigError(f"YAML 的 base_dir 必须是 source_path 的父目录喵! {source_path!r}, {base_dir!r}")
    elif source_path is not None:
        raise ConfigError(f"CLI 没有来源文件，source_path 必须为 None 喵! 现在的值: {source_path!r}")
    if not isinstance(field_path, tuple) or any(type(part) not in (str, int) for part in field_path):
        raise ConfigError(f"field_path 必须为仅含 str 或 int 的元组喵! 现在的值: {field_path!r}")

    source_info: dict[str, str | Path | None] = {
        "kind": kind,
        "source_path": source_path,
        "base_dir": base_dir,
    }
    origins: dict[tuple[str | int, ...], dict[str, str | Path | None]] = {}

    raw_config = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)

    def walk(node: object, current_path: tuple[str | int, ...]) -> None:
        """
        :param node: object, 当前配置节点，可为普通字典、列表或单个值
        :param current_path: tuple[str | int, ...], 当前节点从配置根开始的字段位置

        :return: None, 直接向外层 origins 写入记录

        该函数是collect_origins的内部函数, 用于遍历当前节点及子节点，在外层 origins 中按位置保存 source_info 的副本，空容器也记录
        """
        origins[current_path] = source_info.copy()
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    raise ConfigError(
                        f"配置映射的键必须为 str 喵! 字段位置: {current_path!r}，"
                        f"实际键: {key!r}，来源: {source_info!r}"
                    )
                walk(value, current_path + (key,))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, current_path + (index,))

    walk(raw_config, field_path)
    return origins


def merge_config_sources(
    sources: list[tuple[DictConfig, dict]],
) -> tuple[DictConfig, dict[tuple[str | int, ...], dict[str, str | Path | None]]]:
    """
    :param sources: list[tuple[DictConfig, dict]], 至少一项的列表，每项恰好包含配置和对应来源表，按从低到高的优先级排列

    :return: tuple[DictConfig, dict], 按顺序返回 merged_config 合并配置和 merged_origins 合并来源表

    该函数用于依次用后面的配置覆盖前面的配置，并同步来源；字典逐字段合并、列表整体替换，??? 保留旧具体值时也保留旧来源，输入保持不变
    """
    if not isinstance(sources, list):
        raise ConfigError(f"sources 必须为 list 类型的喵! 现在的 type: {type(sources)}")
    if not sources:
        raise ConfigError("sources 不能为空喵!")

    for source_index, source in enumerate(sources):
        if not isinstance(source, tuple) or len(source) != 2:
            raise ConfigError(f"sources[{source_index}] 必须为二元 tuple 喵! 实际: {source!r}")
        incoming_config, incoming_origins = source
        if not isinstance(incoming_config, DictConfig):
            raise ConfigError(f"sources[{source_index}] 的配置必须为 DictConfig 喵! 现在的 type: {type(incoming_config)}")
        if not isinstance(incoming_origins, dict):
            raise ConfigError(f"sources[{source_index}] 的来源必须为 dict 喵! 现在的type: {incoming_origins}")

        for field_path, source_info in incoming_origins.items():
            if not isinstance(field_path, tuple) or any(type(part) not in (str, int) for part in field_path):
                raise ConfigError(f"sources[{source_index}] 的来源键必须为仅含 str/int 的 tuple 喵! 实际值: {field_path!r}")
            if not isinstance(source_info, dict) or not {"kind", "source_path", "base_dir"}.issubset(source_info):
                raise ConfigError(f"来源记录必须为含 kind/source_path/base_dir 的 dict 喵! 输入序号: {source_index}，字段位置: {field_path!r}")
            kind = source_info["kind"]
            source_path = source_info["source_path"]
            base_dir = source_info["base_dir"]
            if kind not in ("yaml", "cli"):
                raise ConfigError(f"来源 kind 必须为 yaml 或 cli 喵! {field_path!r}: {kind!r}")
            if not isinstance(base_dir, Path) or not base_dir.is_absolute():
                raise ConfigError(f"来源 base_dir 必须为绝对 Path 喵! {field_path!r}: {base_dir!r}")
            if kind == "yaml":
                if not isinstance(source_path, Path) or not source_path.is_absolute():
                    raise ConfigError(f"YAML 来源 source_path 必须为绝对 Path 喵! {field_path!r}: {source_path!r}")
                if source_path.parent != base_dir:
                    raise ConfigError(f"YAML 来源目录与文件父目录不一致喵! {field_path!r}: {source_path!r}, {base_dir!r}")
            elif source_path is not None:
                raise ConfigError(f"CLI 来源 source_path 必须为 None 喵! {field_path!r}")

    initial_config, initial_origins = sources[0]
    merged_config = deepcopy(initial_config)
    merged_origins = deepcopy(initial_origins)
    missing = object()

    def walk(
        old_node: object,
        incoming_node: object,
        result_node: object,
        path: tuple[str | int, ...],
    ) -> None:
        """
        :param old_node: object, 当前字段在本轮合并前的值，字段不存在时为外层 missing 标记
        :param incoming_node: object, 同一字段在本轮新输入中的值，字段不存在时为外层 missing 标记
        :param result_node: object, 同一字段经 OmegaConf 合并后的值，字段不存在时为外层 missing 标记
        :param path: tuple[str | int, ...], 上述三个节点共同对应的字段位置

        :return: None, 直接更新外层 candidate_origins，不修改配置值

        该函数是merge_config_sources的内部函数, 用于对照合并前后及新输入决定来源，未覆盖字段保留原记录，替换子树清除旧记录后写入本轮来源
        """
        if incoming_node is missing:
            return
        if result_node is missing:
            raise ConfigError(f"新输入字段在合并结果中丢失了喵! 字段位置: {path!r}")
        incoming_is_missing_value = isinstance(incoming_node, str) and incoming_node == "???"
        old_is_missing_value = isinstance(old_node, str) and old_node == "???"
        if incoming_is_missing_value and old_node is not missing and not old_is_missing_value:
            if result_node != old_node:
                raise ConfigError(f"??? 合并后未原样保留旧值，无法确定字段来源喵! 字段位置: {path!r}；可能涉及容器插值展开，请人工检查该字段。")
            return

        if isinstance(old_node, dict) and isinstance(incoming_node, dict):
            if not isinstance(result_node, dict):
                raise ConfigError(f"两个映射合并后结果不是 dict 喵! 字段位置: {path!r}")
            for key, incoming_value in incoming_node.items():
                walk(
                    old_node.get(key, missing),
                    incoming_value,
                    result_node.get(key, missing),
                    path + (key,),
                )
            return

        if isinstance(incoming_node, (dict, list)) and result_node != incoming_node:
            raise ConfigError(f"容器合并结果与本次输入不同，无法可靠确定字段来源喵! 字段位置: {path!r}；可能涉及旧容器插值，请人工检查。")

        old_paths = [
            current_path
            for current_path in candidate_origins
            if current_path[:len(path)] == path
        ]
        for current_path in old_paths:
            del candidate_origins[current_path]

        def record_incoming_origins(
            node: object,
            current_path: tuple[str | int, ...],
        ) -> None:
            """
            :param node: object, 合并结果中已采用新输入的节点，可为字典、列表或单个值
            :param current_path: tuple[str | int, ...], 当前节点在合并配置中的位置

            :return: None, 从外层 incoming_origins 读取记录并写入外层 candidate_origins

            该函数为merge_config_sources的walk的内部函数, 用于为替换后的整棵子树逐项登记新来源，当前节点未记录时向父节点查找，再将来源副本写到当前节点位置
            """
            lookup_path = current_path
            while lookup_path not in incoming_origins:
                if not lookup_path:
                    raise ConfigError(f"新输入字段及其父节点均无来源记录喵! 字段位置: {current_path!r}，请检查本次输入的 origins。")
                lookup_path = lookup_path[:-1]
            candidate_origins[current_path] = incoming_origins[lookup_path].copy()
            if isinstance(node, dict):
                for key, value in node.items():
                    record_incoming_origins(value, current_path + (key,))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    record_incoming_origins(value, current_path + (index,))

        record_incoming_origins(result_node, path)

    for source_index, source in enumerate(sources[1:], start=1):
        incoming_config, incoming_origins = source
        try:
            candidate_config = OmegaConf.merge(merged_config, incoming_config)
            old_tree = OmegaConf.to_container(
                merged_config, resolve=False, throw_on_missing=False
            )
            incoming_tree = OmegaConf.to_container(
                incoming_config, resolve=False, throw_on_missing=False
            )
            result_tree = OmegaConf.to_container(
                candidate_config, resolve=False, throw_on_missing=False
            )
        except OmegaConfBaseException as e:
            raise ConfigError(f"配置在合并阶段发生错误了喵! 输入序号: {source_index}，详细报错信息: {e}") from e

        candidate_origins = deepcopy(merged_origins)
        walk(old_tree, incoming_tree, result_tree, ())

        merged_config = candidate_config
        merged_origins = candidate_origins

    return merged_config, merged_origins

def resolve_to_dict(
    merged_config: DictConfig,
    merged_origins: dict[tuple[str | int, ...], dict[str, str | Path | None]],
) -> tuple[dict, dict[tuple[str | int, ...], dict[str, str | Path | None]]]:
    """
    :param merged_config: DictConfig, 已合并但尚未统一求值插值的配置
    :param merged_origins: dict, 合并后的来源表，键为字段位置元组，值为来源记录

    :return: tuple[dict, dict], 按顺序返回解析插值后的普通配置 raw_config 和 merged_origins 的深拷贝

    该函数用于解析 ${...} 并转换成普通字典，拒绝未填的 ??? 和展开为容器的插值，转换失败时查询来源帮助定位错误
    """
    if not isinstance(merged_config, DictConfig):
        raise ConfigError(
            f"merged_config 必须为 DictConfig 类型的喵! "
            f"现在的 type: {type(merged_config)}"
        )

    if not isinstance(merged_origins, dict):
        raise ConfigError(
            f"merged_origins 必须为 dict 类型的喵! "
            f"现在的 type: {type(merged_origins)}"
        )

    def parse_full_key(full_key: object) -> tuple[str | int, ...] | None:
        """
        :param full_key: object, OmegaConf 异常提供的 full_key，通常是类似 components.losses[0].weight 的字符串

        :return: tuple[str | int, ...] | None, 可解析时返回字段位置元组，空值或不支持的格式返回 None

        该函数为resolve_to_dict的内部函数, 用于将点分字段和非负整数方括号下标拆成来源表的键，供外层定位配置转换错误
        """
        if not isinstance(full_key, str) or not full_key:
            return None

        field_path: list[str | int] = []

        for segment in full_key.split("."):
            if not segment:
                return None

            bracket_index = segment.find("[")

            if bracket_index == -1:
                if "]" in segment:
                    return None
                field_path.append(segment)
                continue

            field_name = segment[:bracket_index]
            if not field_name or "]" in field_name:
                return None

            field_path.append(field_name)

            remainder = segment[bracket_index:]

            while remainder:
                if not remainder.startswith("["):
                    return None

                closing_index = remainder.find("]")
                if closing_index <= 1:
                    return None

                index_text = remainder[1:closing_index]
                if not index_text.isdigit():
                    return None

                field_path.append(int(index_text))
                remainder = remainder[closing_index + 1:]

        return tuple(field_path)

    try:
        raw_config = OmegaConf.to_container(
            merged_config,
            resolve=True,
            throw_on_missing=True,
        )

    except OmegaConfBaseException as e:
        full_key = getattr(e, "full_key", None)
        field_path = parse_full_key(full_key)

        source_info = None
        source_lookup_path = None

        if field_path is not None:
            lookup_path = field_path

            while True:
                if lookup_path in merged_origins:
                    candidate_source = merged_origins[lookup_path]

                    if isinstance(candidate_source, dict):
                        source_info = candidate_source
                        source_lookup_path = lookup_path

                    break

                if not lookup_path:
                    break

                lookup_path = lookup_path[:-1]

        if source_info is None:
            source_text = "来源无法定位：没有可用的 full_key，或者 merged_origins 中没有对应字段及父节点记录"

        elif source_info.get("kind") == "yaml":
            source_text = f"来源为 YAML 文件 {source_info.get('source_path')!s}；来源字段位置: {source_lookup_path!r}"

        elif source_info.get("kind") == "cli":
            source_text = f"来源为 CLI 覆盖；来源字段位置: {source_lookup_path!r}"

        else:
            source_text = f"来源记录类型未知: {source_info!r}；来源字段位置: {source_lookup_path!r}"

        raise ConfigError(
            "merged_config 的转换出现问题了喵! "
            f"full_key: {full_key!r}；"
            f"{source_text}；"
            f"原始错误: {e}"
        ) from e

    if not isinstance(raw_config, dict):
        raise ConfigError(f"raw_config 必须为 dict 类型的喵! 现在的 type: {type(raw_config)}")

    unresolved = OmegaConf.to_container(merged_config, resolve=False, throw_on_missing=False)

    def check_container_interpolation(before, after, path):
        """
        :param before: object, 同一节点在插值尚未求值时的值
        :param after: object, 同一节点在插值求值后的值
        :param path: tuple[str | int, ...], 被比较节点的字段位置

        :return: None, 检查通过后结束，不修改两个输入节点

        该函数为resolve_to_dict的内部函数用于递归对照解析前后值，拒绝字符串引用展开成字典或列表，避免无法准确追踪展开子字段的来源
        """
        if isinstance(before, str) and isinstance(after, (dict, list)):
            from .common import config_error
            raise config_error("容器插值无法可靠追踪子字段来源，请显式填写该字典或列表喵~", path)
        if isinstance(before, dict) and isinstance(after, dict):
            for key in before:
                check_container_interpolation(before[key], after[key], path + (key,))
        elif isinstance(before, list) and isinstance(after, list):
            for index, (old, new) in enumerate(zip(before, after)):
                check_container_interpolation(old, new, path + (index,))

    check_container_interpolation(unresolved, raw_config, ())
    return raw_config, deepcopy(merged_origins)

def read_run_header(current_config, current_origins, main_yaml_path) -> tuple[str, Path | None]:
    """
    :param current_config: DictConfig, 当前入口 YAML 与 CLI 合并后的配置，尚未继承训练配置
    :param current_origins: dict, 当前配置的来源表，用于确定 train_config 相对路径的基准
    :param main_yaml_path: Path, 当前主 YAML 文件已经确认有效的绝对路径

    :return: tuple[str, Path | None], 返回规范化运行模式和训练 YAML 的绝对路径，train 模式的第二项为 None

    该函数用于提前读取 run.mode，非训练模式再定位并检查 run.checkpoint.train_config，供后续加载训练配置，不加载 .pt 权重
    """
    from .common import config_error, origin_base
    try:
        mode = OmegaConf.select(current_config, 'run.mode', throw_on_missing=True)
    except OmegaConfBaseException as error:
        raise config_error(f'无法读取 run.mode：{error}', ('run', 'mode')) from error
    if not isinstance(mode, str) or mode.strip().casefold() not in ('train', 'test', 'inference', 'visualize'):
        raise config_error('run.mode 必须显式填写 train/test/inference/visualize。', ('run', 'mode'))
    mode = mode.strip().casefold()
    if mode == 'train':
        return mode, None
    path = ('run', 'checkpoint', 'train_config')
    try:
        value = OmegaConf.select(current_config, 'run.checkpoint.train_config', throw_on_missing=True)
    except OmegaConfBaseException as error:
        raise config_error(f'无法解析训练来源入口，该字段只能引用当前配置中可解析的值：{error}', path) from error
    if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value.strip()):
        raise config_error('run.checkpoint.train_config 必须为非空训练 YAML 路径。', path)
    candidate = Path(value.strip()) if isinstance(value, str) else value
    base_dir = origin_base(current_origins, path, main_yaml_path.parent)
    try:
        candidate = resolve_config_path(candidate if candidate.is_absolute() else base_dir / candidate)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise config_error(f'训练配置来源无效：{error}', path) from error
    if candidate == main_yaml_path:
        raise config_error('非训练配置不能将自身作为训练来源；训练源不允许递归继承。', path)
    return mode, candidate


def load_training_projection(train_yaml_path: Path, mode: str) -> tuple[DictConfig, dict]:
    """
    :param train_yaml_path: Path, run.checkpoint.train_config 指向的有效训练 YAML 绝对路径
    :param mode: str, 当前非训练运行模式，只允许 test、inference 或 visualize

    :return: tuple[DictConfig, dict], 返回允许继承的配置片段及对应来源表，其中 model.init_from 被设为 None

    该函数用于读取 version 为 2 的训练配置，提取 version、experiment、task、tokenizer、model，test 额外提取 losses，并记录禁用训练初始化的原因
    """
    from .common import config_error, nearest_origin
    if mode not in ('test', 'inference', 'visualize'):
        raise config_error('只有非训练模式可以加载训练投影。', ('run', 'mode'))
    source = load_yaml(train_yaml_path)
    origins = collect_origins(source, kind='yaml', source_path=train_yaml_path, base_dir=train_yaml_path.parent)
    try:
        raw, origins = resolve_to_dict(source, origins)
    except ConfigError as error:
        error.config_origins = origins
        raise

    def source_error(message, path):
        """
        :param message: str, 对训练来源配置发现的问题说明
        :param path: tuple[str | int, ...], 问题在训练配置中的字段位置

        :return: ConfigError, 附带字段位置和训练来源 origins 的异常对象，由调用方抛出

        该函数为load_training_projection的内部函数, 用于读取外层 train_yaml_path 和 origins 构造训练来源异常，避免错误被归到当前测试或推理 YAML
        """
        error = config_error(f'训练来源 {train_yaml_path}：{message}', path)
        error.config_origins = origins
        return error

    if type(raw.get('version')) is not int or raw['version'] != 2:
        raise source_error('要求 version: 2，请先迁移旧训练快照。', ('version',))
    run = raw.get('run')
    if not isinstance(run, dict) or not isinstance(run.get('mode'), str) or run['mode'].strip().casefold() != 'train':
        raise source_error('run.mode 必须为 train，不递归加载其他来源。', ('run', 'mode'))
    for key in ('experiment', 'task', 'components'):
        if not isinstance(raw.get(key), dict):
            raise source_error(f'{key} 必须为字典；旧版平铺组件需要先迁移。', (key,))
    for key in ('tokenizer', 'model'):
        if not isinstance(raw['components'].get(key), dict):
            raise source_error(f'components.{key} 必须为字典。', ('components', key))
    selected = {key: deepcopy(raw[key]) for key in ('version', 'experiment', 'task')}
    selected['components'] = {key: deepcopy(raw['components'][key]) for key in ('tokenizer', 'model')}
    if mode == 'test':
        if 'losses' not in raw['components']:
            raise source_error('测试需要继承训练 components.losses 喵! 没有在train_config中找到该字段喵!', ('components', 'losses'))
        selected['components']['losses'] = deepcopy(raw['components']['losses'])
    selected['components']['model']['init_from'] = None
    projected_origins = {}

    def copy_origins(node, path):
        """
        :param node: object, 已选中继承的节点，可为字典、列表或单个值
        :param path: tuple[str | int, ...], 当前节点在配置中的位置

        :return: None, 向外层 projected_origins 写入来源副本

        该函数用于遍历继承片段，从外层训练 origins 查询当前节点或最近父节点来源，深拷贝后登记在对应位置
        """
        projected_origins[path] = deepcopy(nearest_origin(origins, path))
        children = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, value in children:
            copy_origins(value, path + (key,))
    copy_origins(selected, ())
    projected_origins[('components', 'model', 'init_from')]['transformation'] = '由继承政策禁用训练初始化'
    return OmegaConf.create(selected), projected_origins


def merge_inherited_config(inherited_config, inherited_origins, current_config, current_origins, mode, task_specs) -> tuple[DictConfig, dict]:
    """
    :param inherited_config: DictConfig, 从训练 YAML 提取的允许继承的配置片段
    :param inherited_origins: dict, 继承片段对应的来源表
    :param current_config: DictConfig, 当前非训练 YAML 与 CLI 合并后的配置
    :param current_origins: dict, 当前配置对应的来源表
    :param mode: str, read_run_header 已确定的 test、inference 或 visualize 模式
    :param task_specs: dict, 按任务规范名保存配置规则的表，供比较 task.type 时确认任务已有解析规则，别名由任务注册表解析

    :return: tuple[DictConfig, dict], 返回当前输入覆盖继承片段后的 effective 配置及 origins 来源表

    该函数用于以当前配置为高优先级合并，禁止重新声明 components，并检查版本、任务和运行模式保持符合继承规则
    """
    from .common import config_error
    from .schema_registry import resolve_task_key
    current = OmegaConf.to_container(current_config, resolve=False, throw_on_missing=False)
    if 'components' in current:
        raise config_error('非训练配置禁止声明 components，包括相同值；请使用对应训练快照。', ('components',))
    effective, origins = merge_config_sources([
        (inherited_config, inherited_origins),
        (current_config, current_origins),
    ])
    try:
        version = OmegaConf.select(effective, 'version', throw_on_missing=True)
        if type(version) is not int or version != 2:
            raise config_error('非训练 version 必须与训练来源一致且为严格整数 2。', ('version',))
        if 'task' in current:
            if not isinstance(current['task'], dict):
                raise config_error('task 必须为字典。', ('task',))
            if 'type' in current['task']:
                if current['task']['type'] is None or current['task']['type'] == '???':
                    raise config_error('显式 task.type 不能为 null 或 ???。', ('task', 'type'))
                actual = OmegaConf.select(effective, 'task.type', throw_on_missing=True)
                source = OmegaConf.select(inherited_config, 'task.type', throw_on_missing=True)
                if resolve_task_key(actual, task_specs) != resolve_task_key(source, task_specs):
                    raise config_error('非训练 task.type 必须与训练来源一致，不能更换 checkpoint 的任务。', ('task', 'type'))
        if 'version' in current and current['version'] == '???':
            raise config_error('显式 version 不能用 ??? 代替版本号。', ('version',))
        actual_mode = OmegaConf.select(effective, 'run.mode', throw_on_missing=True)
        if not isinstance(actual_mode, str) or actual_mode.strip().casefold() != mode:
            raise config_error('继承后的 run.mode 与读取入口时不同。', ('run', 'mode'))
    except OmegaConfBaseException as error:
        raise config_error(f'继承覆盖字段无法求值：{error}') from error
    return effective, origins
