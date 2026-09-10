from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from metrics.registry import METRIC_REGISTRY
from metrics.core.collection import MetricCollection
from metrics.output.writer import MetricWriter
from tasks.MLM_PreTrain.mlm import MLMTask
from loaders.mlm import MLMDataset, MLMCollator

from torch.utils.data import DataLoader

def build_mlm_test(config_dict, test_metrics_path, logger=None):
    train_config_path = config_dict["run"]["checkpoint"]["train_config"]
    from parse.parser import parse_config
    train_config, parse_context, compiled_schema = parse_config(
        config_path=train_config_path,
        # PARSE_REFACTOR_DEMO: 这里解析的是 checkpoint 引用的训练配置，
        # 因此必须按训练 task 的 mode 校验，而不是按当前 Test 入口校验。
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
    test_path = dataset_splits_config["test"]
    dataset_params_config = dataset_config["params"]
    max_seq_length = dataset_params_config["max_seq_length"]
    mlm_prob = dataset_params_config["mlm_prob"]
    # 实例化训练集
    test_dataset = MLMDataset(
        data_file_path=test_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split_name="test",
        progress_interval=10000,
        read_interval=10000,
        logger=logger,
    )
    # 实例化collator
    mlm_collator = MLMCollator(
        tokenizer=tokenizer,
        mlm_prob=mlm_prob,
    )

    # 接收dataloader参数
    loaders_config = data_config["loaders"]
    dataloader_config = loaders_config["test"]
    batch_size = dataloader_config["batch_size"]
    shuffle = dataloader_config["shuffle"]
    num_workers = dataloader_config["num_workers"]
    pin_memory = dataloader_config["pin_memory"]
    drop_last = dataloader_config["drop_last"]
    # 实例化dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
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
    model_init_from = model_config["init_from"]
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

    # 接收losses参数
    LOSS_CANONICAL_MAP = LOSS_REGISTRY.canonical_map
    losses_config = config_dict["losses"]
    loss_list = []
    for loss_config in losses_config:
        loss_dict = {}
        loss_name = loss_config["name"]
        loss_type = loss_config["type"].strip().casefold()
        loss_weight = loss_config["weight"]
        loss_params = loss_config["params"]
        # 实例化loss
        loss_class = LOSS_CANONICAL_MAP[loss_type]
        loss = loss_class(**loss_params)
        loss_dict["name"] = loss_name
        loss_dict["loss_fn"] = loss
        loss_dict["weight"] = loss_weight
        loss_list.append(loss_dict)
    # 实例化WeightedLoss
    loss_fn = WeightedLoss(loss_list)

    # 接收evaluation字段
    METRIC_CANONICAL_MAP = METRIC_REGISTRY.canonical_map
    evaluation_config = config_dict["evaluation"]
    metrics_config = evaluation_config["metrics"]
    test_metrics_config = metrics_config["test"]
    test_metric_list = []
    for test_metric in test_metrics_config:
        test_metric_type = test_metric["type"].strip().casefold()
        test_metric_params = test_metric["params"]
        test_metric_class = METRIC_CANONICAL_MAP[test_metric_type]
        test_metric_list.append(test_metric_class(**test_metric_params))
    test_metric_collection = MetricCollection(test_metric_list)

    base_fieldnames = ["global_step"]
    loss_fieldnames = ["avg_loss"]
    for loss_dict in loss_list:
        loss_name = loss_dict["name"]
        loss_fieldnames.append(f"{loss_name}_avg_raw")
        loss_fieldnames.append(f"{loss_name}_avg_weighted_loss")
    test_metric_fieldnames = []
    for metric in test_metric_list:
        test_metric_fieldnames.extend(metric.output_fields())
    test_fieldnames = base_fieldnames + loss_fieldnames + test_metric_fieldnames
    test_metrics_writer = MetricWriter(test_metrics_path, test_fieldnames)

    # 实例化task
    mlm_task = MLMTask(
        model=model,
        loss_fn=loss_fn,
    )

    # 组装测试字典
    test_params_dict = {
        "task": mlm_task,
        "test_dataloader": test_dataloader,
        "test_metric_collection": test_metric_collection,
        "test_metrics_writer": test_metrics_writer,
        "test_metrics_path": test_metrics_path,
    }

    return test_params_dict
