from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from loaders.mlm import MLMDataset, MLMCollator
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from tasks.MLM_PreTrain.mlm import MLMTask
from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
from metrics.registry import METRIC_REGISTRY
from metrics.core.collection import MetricCollection
from metrics.output.writer import MetricWriter
from Trainer.trainer import Trainer
from Trainer.checkpoint import load_checkpoint, load_checkpoint_state

from torch.utils.data import DataLoader

def build_mlm_pretrain(config_dict, device_summary, resume=None, experiment_paths=None, logger=None):
    # 接收tokenizer参数
    tokenizer_config = config_dict["tokenizer"]
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
    dataset_params_config = dataset_config["params"]
    train_path = dataset_splits_config["train"]
    max_seq_length = dataset_params_config["max_seq_length"]
    mlm_prob = dataset_params_config["mlm_prob"]
    # 实例化训练集
    train_dataset = MLMDataset(
        data_file_path=train_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split_name="train",
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
    train_loaders_config = loaders_config["train"]
    train_batch_size = train_loaders_config["batch_size"]
    train_shuffle = train_loaders_config["shuffle"]
    train_num_workers = train_loaders_config["num_workers"]
    train_pin_memory = train_loaders_config["pin_memory"]
    train_drop_last = train_loaders_config["drop_last"]
    valid_loaders_config = loaders_config["valid"]
    valid_batch_size = valid_loaders_config["batch_size"]
    valid_shuffle = valid_loaders_config["shuffle"]
    valid_num_workers = valid_loaders_config["num_workers"]
    valid_pin_memory = valid_loaders_config["pin_memory"]
    valid_drop_last = valid_loaders_config["drop_last"]
    test_loaders_config = loaders_config["test"]
    test_batch_size = test_loaders_config["batch_size"]
    test_shuffle = test_loaders_config["shuffle"]
    test_num_workers = test_loaders_config["num_workers"]
    test_pin_memory = test_loaders_config["pin_memory"]
    test_drop_last = test_loaders_config["drop_last"]
    # 实例化dataloader
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = train_batch_size,
        shuffle = train_shuffle,
        num_workers = train_num_workers,
        pin_memory = train_pin_memory,
        drop_last = train_drop_last,
        collate_fn = mlm_collator,
    )

    # 接收model参数
    model_config = config_dict["model"]
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
    losses_config = config_dict["losses"]
    loss_list = []
    for loss_config in losses_config:
        loss_dict = {}
        loss_name = loss_config["name"]
        loss_type = loss_config["type"].strip().casefold()
        loss_weight = loss_config["weight"]
        loss_params = loss_config["params"]
        # 实例化loss
        LOSS_CANONICAL_MAP = LOSS_REGISTRY.canonical_map
        loss_class = LOSS_CANONICAL_MAP[loss_type]
        loss = loss_class(**loss_params)
        loss_dict["name"] = loss_name
        loss_dict["loss_fn"] = loss
        loss_dict["weight"] = loss_weight
        loss_list.append(loss_dict)
    # 实例化WeightedLoss
    loss_fn = WeightedLoss(loss_list)

    # 接收evaluation字段
    evaluation_config = config_dict["evaluation"]
    test_after_train = evaluation_config["test_after_train"]
    metrics_config = evaluation_config["metrics"]
    train_metrics_config = metrics_config["train"]
    train_metric_list = []
    METRIC_CANONICAL_MAP = METRIC_REGISTRY.canonical_map
    for train_metric in train_metrics_config:
        train_metric_type = train_metric["type"].strip().casefold()
        train_metric_params = train_metric["params"]
        train_metric_class = METRIC_CANONICAL_MAP[train_metric_type]
        train_metric_list.append(train_metric_class(**train_metric_params))
    train_metric_collection = MetricCollection(train_metric_list)
    valid_metrics_config = metrics_config["valid"]
    valid_metric_list = []
    for valid_metric in valid_metrics_config:
        valid_metric_type = valid_metric["type"].strip().casefold()
        valid_metric_params = valid_metric["params"]
        valid_metric_class = METRIC_CANONICAL_MAP[valid_metric_type]
        valid_metric_list.append(valid_metric_class(**valid_metric_params))
    valid_metric_collection = MetricCollection(valid_metric_list)
    test_metrics_config = metrics_config["test"]
    test_metric_list = []
    for test_metric in test_metrics_config:
        test_metric_type = test_metric["type"].strip().casefold()
        test_metric_params = test_metric["params"]
        test_metric_class = METRIC_CANONICAL_MAP[test_metric_type]
        test_metric_list.append(test_metric_class(**test_metric_params))
    test_metric_collection = MetricCollection(test_metric_list)

    # 构建metric的csv文件保存配置
    metrics_dir = experiment_paths["metrics_dir"]
    train_metrics_path = metrics_dir / "train_metrics.csv"
    valid_metrics_path = metrics_dir / "valid_metrics.csv"
    test_metrics_path = metrics_dir / "test_metrics.csv"

    base_fieldnames = ["global_step"]
    loss_fieldnames = ["avg_loss"]
    for loss_dict in loss_list:
        loss_name = loss_dict["name"]
        loss_fieldnames.append(f"{loss_name}_avg_raw")
        loss_fieldnames.append(f"{loss_name}_avg_weighted_loss")
    interval_loss_fieldnames = ["interval_avg_loss"]
    for loss_dict in loss_list:
        loss_name = loss_dict["name"]
        interval_loss_fieldnames.append(f"{loss_name}_interval_avg_raw")
        interval_loss_fieldnames.append(f"{loss_name}_interval_avg_weighted_loss")
    train_metric_fieldnames = []
    for metric in train_metric_list:
        train_metric_fieldnames.extend(metric.output_fields())
    valid_metric_fieldnames = []
    for metric in valid_metric_list:
        valid_metric_fieldnames.extend(metric.output_fields())
    train_fieldnames = base_fieldnames + ['lr'] + loss_fieldnames + interval_loss_fieldnames + train_metric_fieldnames
    valid_fieldnames = base_fieldnames + loss_fieldnames + valid_metric_fieldnames
    train_metrics_writer = MetricWriter(train_metrics_path, train_fieldnames)
    valid_metrics_writer = MetricWriter(valid_metrics_path, valid_fieldnames)
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

    # 接收optimization参数
    optimization_config = config_dict["optimization"]
    # 接收optimizer参数
    optimizer_config = optimization_config["optimizer"]
    optimizer_type = optimizer_config["type"].strip().casefold()
    optimizer_params_config = optimizer_config["params"]
    lr = optimizer_params_config["lr"]
    weight_decay = optimizer_params_config["weight_decay"]
    betas = optimizer_params_config["betas"]
    eps = optimizer_params_config["eps"]
    betas = tuple(betas)
    # 获得optimizer的类
    OPTIMIZER_CANONICAL_MAP = OPTIMIZER_REGISTRY.canonical_map
    optimizer_class = OPTIMIZER_CANONICAL_MAP[optimizer_type]
    # 实例化optimizer
    optimizer = optimizer_class(
        mlm_task.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    # 接收scheduler参数
    scheduler_config = optimization_config["scheduler"]
    scheduler_type = scheduler_config["type"].strip().casefold()
    scheduler_params_config = scheduler_config["params"]
    interval = scheduler_config["interval"]
    # 获得scheduler的类
    SCHEDULER_CANONICAL_MAP = SCHEDULER_REGISTRY.canonical_map
    scheduler_class = SCHEDULER_CANONICAL_MAP[scheduler_type]
    # 实例化scheduler
    scheduler = scheduler_class(
        optimizer=optimizer,
        **scheduler_params_config,
    )

    # 接收runtime参数
    runtime_config = config_dict["runtime"]
    device_config = runtime_config["device"]
    device_type = device_config["type"].strip().casefold()
    device_index = device_config["index"]
    device_count = device_config["count"]

    # 接收loop参数
    loop_config = optimization_config["loop"]
    total_steps = loop_config["total_steps"]
    log_every_steps = loop_config["log_every_steps"]
    accumulation_steps = loop_config["accumulation_steps"]

    # 接收validation参数
    validation_config = optimization_config["validation"]
    validation_enabled = validation_config["enabled"]

    # 实例化trainer
    trainer = Trainer(
        optimizer=optimizer,
        scheduler=scheduler,
        device=device_type,
        device_summary=device_summary,
        task=mlm_task,
        logger=logger,
        train_metric_collection=train_metric_collection,
        valid_metric_collection=valid_metric_collection,
        train_metrics_writer=train_metrics_writer,
        valid_metrics_writer=valid_metrics_writer
    )

    if resume is not None:
        checkpoint = load_checkpoint(resume)
        load_checkpoint_state(
            checkpoint=checkpoint,
            task=mlm_task,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer=trainer,
        )
        if total_steps <= trainer.global_step:
            raise ValueError(f"resume_path: {resume}"
                             f"total_steps: {total_steps}"
                             f"trainer.global_step: {trainer.global_step}"
                             f"total_steps 必须大于 trainer.global_step 喵!")

    # validation_enabled 为True时接收valid参数，False时为None
    valid_dataloader = None
    validation_every_steps = None
    early_stopping_enabled = False
    early_stopping_patience = None
    if validation_enabled:
        validation_every_steps = validation_config["every_steps"]
        validation_early_stopping_config = validation_config["early_stopping"]
        early_stopping_enabled = validation_early_stopping_config["enabled"]
        early_stopping_patience = validation_early_stopping_config["patience"]
        checkpoint_config = validation_config["checkpoint"]
        checkpoint_save_every_rounds = checkpoint_config["save_every_rounds"]
        valid_path = dataset_splits_config["valid"]
        # 实例化dataset
        valid_dataset = MLMDataset(
            data_file_path=valid_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split_name="valid",
            progress_interval=10000,
            read_interval=10000,
            logger=logger,
        )
        # 实例化dataloader
        valid_dataloader = DataLoader(
            dataset = valid_dataset,
            batch_size = valid_batch_size,
            shuffle = valid_shuffle,
            num_workers = valid_num_workers,
            pin_memory = valid_pin_memory,
            drop_last = valid_drop_last,
            collate_fn = mlm_collator,
        )

    # test.enabled 为True时接收valid参数，False时为None
    test_dataloader = None
    if test_after_train:
        test_path = dataset_splits_config["test"]
        # 实例化dataset
        test_dataset = MLMDataset(
            data_file_path=test_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split_name="test",
            progress_interval=10000,
            read_interval=10000,
            logger=logger,
        )
        # 实例化dataloader
        test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size = test_batch_size,
            shuffle = test_shuffle,
            num_workers = test_num_workers,
            pin_memory = test_pin_memory,
            drop_last = test_drop_last,
            collate_fn = mlm_collator,
        )

    # 组装参数字典
    train_params_dict = {
        "trainer": trainer,
        "task": mlm_task,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "experiment_paths": experiment_paths,
        "train_metric_collection": train_metric_collection,
        "valid_metric_collection": valid_metric_collection,
        "test_metric_collection": test_metric_collection,
        "train_metrics_writer": train_metrics_writer,
        "valid_metrics_writer": valid_metrics_writer,
        "test_metrics_writer": test_metrics_writer,
        "train_metrics_path": train_metrics_path,
        "valid_metrics_path": valid_metrics_path,
        "test_metrics_path": test_metrics_path,
        "fit_kwargs": {
            "total_steps": total_steps,
            "log_every_steps": log_every_steps,
            "accumulation_steps": accumulation_steps,
            "validation_enabled": validation_enabled,
            "validation_every_steps": validation_every_steps,
            "valid_dataloader": valid_dataloader,
            "early_stopping_patience": early_stopping_patience,
        }
    }

    if validation_enabled:
        checkpoints_dir = experiment_paths["checkpoints_dir"]
        train_params_dict["fit_kwargs"]["checkpoint_save_every_rounds"] = checkpoint_save_every_rounds
        train_params_dict["fit_kwargs"]["checkpoints_dir"] = checkpoints_dir


    return train_params_dict
