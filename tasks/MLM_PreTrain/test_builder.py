import pathlib
import yaml

from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from metrics.registry import METRIC_REGISTRY
from metrics.collection import MetricCollection
from metrics.writer import MetricWriter
from tasks.MLM_PreTrain.mlm import MLMTask
from loaders.mlm import MLMDataset, MLMCollator

from torch.utils.data import DataLoader

def build_mlm_test(config_dict, test_metrics_path, logger=None):
    train_config_path = config_dict["checkpoint"]["config_path"]
    from parse.parser import load_config
    train_config = load_config(train_config_path)
    # 接收tokenizer参数
    tokenizer_config = train_config["tokenizer"]
    tokenizer_type = tokenizer_config["type"].strip().casefold()
    vocab_file_path = tokenizer_config["vocab_file_path"]
    do_lower_case = tokenizer_config["do_lower_case"]
    strip_accents = tokenizer_config["strip_accents"]
    max_input_chars_per_word = tokenizer_config["max_input_chars_per_word"]
    # 获取tokenizer的类
    tokenizer_class = TOKENIZER_REGISTRY[tokenizer_type]
    # 实例化tokenizer
    tokenizer = tokenizer_class(
        vocab_file_path=vocab_file_path,
        do_lower_case=do_lower_case,
        strip_accents=strip_accents,
        max_input_chars_per_word=max_input_chars_per_word,
    )

    # 接收dataset参数
    dataset_config = config_dict["dataset"]
    test_path = dataset_config["test_path"]
    max_seq_length = dataset_config["max_seq_length"]
    mlm_prob = dataset_config["mlm_prob"]
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
    dataloader_config = config_dict["dataloader"]
    batch_size = dataloader_config["batch_size"]
    num_workers = dataloader_config["num_workers"]
    pin_memory = dataloader_config["pin_memory"]
    # 实例化dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mlm_collator,
    )

    # 接收model参数
    model_config = train_config["model"]
    model_name = model_config["name"].strip().casefold()
    hidden_size = model_config["hidden_size"]
    max_position_embeddings = model_config["max_position_embeddings"]
    type_vocab_size = model_config["type_vocab_size"]
    num_heads = model_config["num_heads"]
    intermediate_size = model_config["intermediate_size"]
    num_layers = model_config["num_layers"]
    embedding_dropout = model_config["embedding_dropout"]
    attention_dropout = model_config["attention_dropout"]
    hidden_dropout = model_config["hidden_dropout"]
    layer_norm_eps = model_config["layer_norm_eps"]
    # 从tokenizer中获取其他所需参数
    vocab_size = len(tokenizer)
    padding_idx = tokenizer.pad_token_id
    # 获得model的类
    model_class = MODEL_REGISTRY[model_name]
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
    losses_config = config_dict["losses"]
    loss_list = []
    for loss_config in losses_config:
        loss_dict = {}
        loss_name = loss_config["name"]
        loss_type = loss_config["type"].strip().casefold()
        loss_weight = loss_config["weight"]
        loss_params = loss_config["params"]
        # 实例化loss
        loss_class = LOSS_REGISTRY[loss_type]
        loss = loss_class(**loss_params)
        loss_dict["name"] = loss_name
        loss_dict["loss_fn"] = loss
        loss_dict["weight"] = loss_weight
        loss_list.append(loss_dict)
    # 实例化WeightedLoss
    loss_fn = WeightedLoss(loss_list)

    # 接收metrics字段
    metrics_config = config_dict["metrics"]
    test_metrics_config = metrics_config["test"]
    test_metric_list = []
    for test_metric in test_metrics_config:
        test_metric_type = test_metric["type"].strip().casefold()
        test_metric_params = test_metric["params"]
        test_metric_class = METRIC_REGISTRY[test_metric_type]
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
