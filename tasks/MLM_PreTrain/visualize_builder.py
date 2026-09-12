from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from tasks.MLM_PreTrain.mlm import MLMTask
from loaders.mlm import MLMDataset, MLMCollator
from visualize.registry import VISUALIZER_REGISTRY
# AI-CHANGE: 可视化 Builder 内部装载权重(AI 编写, 待人工审查)。
# 与 Tester/Inference 的 runner 职责划分不同：可视化 Builder 已经从
# checkpoint.train_config 恢复了模型结构，权重路径来自同一份 run.checkpoint，
# 因此在 Builder 内一次性完成"建结构 + 装权重"，入口不再重复读 checkpoint。
from Trainer.checkpoint import load_checkpoint

from torch.utils.data import DataLoader

# AI-CHANGE: MLM 可视化 Builder(AI 编写, 待人工审查)。
# 只组装可视化需要的四件套(tokenizer/model/task/dataloader/visualizers),
# 不组装 losses/metrics/optimizer；模型结构从 checkpoint.train_config 恢复，
# 权重在 Builder 内从 run.checkpoint.path 装载；数据集取 data.splits.visualize，
# loader 读 loaders.visualize。
def build_mlm_visualize(config_dict, logger=None):
    if not isinstance(config_dict, dict):
        raise ValueError(f"config_dict 必须为dict类型的喵! 现在的type: {type(config_dict)}")
    if not config_dict:
        raise ValueError(f"config_dict 不能为空喵!")
    if not "run" in config_dict:
        raise ValueError(f"config_dict 必须包含 run 字段喵!")
    if not "checkpoint" in config_dict["run"]:
        raise ValueError(f"config_dict['run'] 必须包含 checkpoint 字段喵!")
    train_config_path = config_dict["run"]["checkpoint"]["train_config"]
    from parse.parser import parse_config
    train_config, parse_context, compiled_schema = parse_config(
        config_path=train_config_path,
        expected_mode='train',
        validate_runtime=False,
    )
    # 接收tokenizer参数
    tokenizer_config = train_config["tokenizer"]
    tokenizer_type = tokenizer_config["type"].strip().casefold()
    tokenizer_params_config = tokenizer_config["params"]
    vocab_file_path = tokenizer_params_config["vocab_file_path"]
    do_lower_case = tokenizer_params_config["do_lower_case"]
    strip_accents = tokenizer_params_config["strip_accents"]
    max_input_chars_per_word = tokenizer_params_config["max_input_chars_per_word"]
    # 获取tokenizer的类
    TOKENIZER_CANONICAL_MAP = TOKENIZER_REGISTRY.canonical_map
    tokenizer_class = TOKENIZER_CANONICAL_MAP[tokenizer_type]
    # 实例化tokenizer
    tokenizer = tokenizer_class(
        vocab_file_path=vocab_file_path,
        do_lower_case=do_lower_case,
        strip_accents=strip_accents,
        max_input_chars_per_word=max_input_chars_per_word,
    )

    # 接收dataset参数
    data_config = config_dict["data"]
    dataset_config = data_config["dataset"]
    dataset_splits_config = dataset_config["splits"]
    visualize_path = dataset_splits_config["visualize"]
    dataset_params_config = dataset_config["params"]
    max_seq_length = dataset_params_config["max_seq_length"]
    mlm_prob = dataset_params_config["mlm_prob"]
    # 实例化数据集
    visualize_dataset = MLMDataset(
        data_file_path=visualize_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split_name="visualize",
        progress_interval=10000,
        read_interval=10000,
        logger=logger,
    )
    # 实例化collator
    mlm_collator = MLMCollator(
        tokenizer=tokenizer,
        mlm_prob=mlm_prob,
    )

    # 接收loaders参数
    loaders_config = data_config["loaders"]
    dataloader_config = loaders_config["visualize"]
    batch_size = dataloader_config["batch_size"]
    shuffle = dataloader_config["shuffle"]
    num_workers = dataloader_config["num_workers"]
    pin_memory = dataloader_config["pin_memory"]
    drop_last = dataloader_config["drop_last"]
    # 实例化dataloader
    visualize_dataloader = DataLoader(
        dataset=visualize_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mlm_collator,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    # 接收model参数
    model_config = train_config["model"]
    model_type = model_config["type"].strip().casefold()
    model_params_config = model_config["params"]
    hidden_size = model_params_config["hidden_size"]
    max_position_embeddings = model_params_config["max_position_embeddings"]
    type_vocab_size = model_params_config["type_vocab_size"]
    num_heads = model_params_config["num_heads"]
    intermediate_size = model_params_config["intermediate_size"]
    num_layers = model_params_config["num_layers"]
    embedding_dropout = model_params_config["embedding_dropout"]
    attention_dropout = model_params_config["attention_dropout"]
    hidden_dropout = model_params_config["hidden_dropout"]
    layer_norm_eps = model_params_config["layer_norm_eps"]
    # 从tokenizer中获取其他所需参数
    vocab_size = len(tokenizer)
    padding_idx = tokenizer.pad_token_id
    # 获得model的类
    MODEL_CANONICAL_MAP = MODEL_REGISTRY.canonical_map
    model_class = MODEL_CANONICAL_MAP[model_type]
    # 实例化model
    model = model_class(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        padding_idx=padding_idx,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        embedding_dropout=embedding_dropout,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        layer_norm_eps=layer_norm_eps,
    )

    # 实例化task(只做权重容器, loss_fn 为None)
    mlm_task = MLMTask(
        model=model,
        loss_fn=None,
    )

    # AI-CHANGE: Builder 内装载 checkpoint 权重(AI 编写, 待人工审查)。
    # .pt 存的是 task_state(键带 model. 前缀),正好对应 MLMTask 的结构；
    # test/inference 在 runner 里装是因为入口不恢复结构，可视化 Builder 已恢复
    # 结构，同一份 run.checkpoint 在此一次装完，入口不再重复读。
    checkpoint_path = config_dict["run"]["checkpoint"]["path"]
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    task_state = checkpoint["task_state"]
    mlm_task.load_state_dict(task_state, strict=True)
    if logger is None:
        print(f"checkpoint 加载成功喵! checkpoint_path: {checkpoint_path}")
    else:
        logger.info(f"checkpoint 加载成功喵! checkpoint_path: {checkpoint_path}")

    # 接收visualize参数
    if not "visualize" in config_dict:
        raise ValueError(f"config_dict 必须包含 visualize 字段喵!")
    visualize_config = config_dict["visualize"]
    if not "methods" in visualize_config:
        raise ValueError(f"config_dict['visualize'] 必须包含 methods 字段喵!")
    methods_config = visualize_config["methods"]
    if not isinstance(methods_config, list):
        raise ValueError(f"methods 必须为list类型的喵! 现在的type: {type(methods_config)}")
    if not methods_config:
        raise ValueError(f"methods 不能为空喵!")
    visualizers = []
    visualizer_names = []
    for method_config in methods_config:
        if not isinstance(method_config, dict):
            raise ValueError(f"method_config 必须为dict类型的喵! 现在的type: {type(method_config)}")
        if not "type" in method_config:
            raise ValueError(f"method_config 必须包含 type 字段喵!")
        if not isinstance(method_config["type"], str):
            raise ValueError(f"method_config['type'] 必须为str类型的喵! 现在的type: {type(method_config['type'])}")
        if not method_config["type"].strip():
            raise ValueError(f"method_config['type'] 不能为空喵!")
        # AI-CHANGE: 先 resolve_key 再 get(AI 编写, 待人工审查)。
        # canonical_map 只有规范键(如 attentionheatmap),yaml 里写的是别名
        # (如 attention_heatmap),直接查表会 KeyError,必须走别名归一化。
        canonical_key = VISUALIZER_REGISTRY.resolve_key(method_config["type"])
        visualizer_class = VISUALIZER_REGISTRY.get(canonical_key)
        method_params = method_config.get("params", {})
        if method_params is None:
            method_params = {}
        if not isinstance(method_params, dict):
            raise ValueError(f"method_params 必须为dict类型的喵! 现在的type: {type(method_params)}")
        method_params = dict(method_params)
        losses_config = method_params.pop("losses", None)
        if losses_config is not None:
            if not isinstance(losses_config, list) or not losses_config:
                raise ValueError("method_params['losses'] 必须为非空list类型的喵!")
            loss_list = []
            for loss_config in losses_config:
                if not isinstance(loss_config, dict):
                    raise TypeError("loss_config 必须为dict类型的喵!")
                loss_name = loss_config["name"]
                loss_type = loss_config["type"]
                loss_weight = loss_config["weight"]
                loss_params = loss_config.get("params") or {}
                loss_key = LOSS_REGISTRY.resolve_key(loss_type)
                loss_class = LOSS_REGISTRY.get(loss_key)
                loss_list.append({
                    "name": loss_name,
                    "loss_fn": loss_class(**loss_params),
                    "weight": loss_weight,
                })
            method_params["loss_fn"] = WeightedLoss(loss_list)
        visualizer = visualizer_class(**method_params)
        visualizers.append(visualizer)
        visualizer_names.append(canonical_key)
    if not visualizers:
        raise ValueError(f"visualizers 不能为空喵!")

    # 组装可视化字典
    visualize_params_dict = {
        "task": mlm_task,
        "model": model,
        "tokenizer": tokenizer,
        "visualize_dataloader": visualize_dataloader,
        "visualizers": visualizers,
        "visualizer_names": visualizer_names,
    }

    return visualize_params_dict
