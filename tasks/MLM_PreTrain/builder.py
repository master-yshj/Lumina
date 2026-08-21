from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from loaders.mlm import MLMDataset, MLMCollator
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from tasks.MLM_PreTrain.mlm import MLMTask
from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
from metrics.registry import METRIC_REGISTRY
from metrics.collection import MetricCollection
from metrics.writer import MetricWriter
from Trainer.trainer import Trainer
from Trainer.checkpoint import load_checkpoint, load_checkpoint_state

from torch.utils.data import DataLoader

def build_mlm_pretrain(config_dict, device_summary, resume=None, experiment_paths=None, logger=None):
    # 接收tokenizer参数
    tokenizer_config = config_dict["tokenizer"]
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
    train_path = dataset_config["train_path"]
    max_seq_length = dataset_config["max_seq_length"]
    mlm_prob = dataset_config["mlm_prob"]
    # 实例化训练集
    train_dataset = MLMDataset(
        data_file_path=train_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    # 实例化collator
    mlm_collator = MLMCollator(
        tokenizer=tokenizer,
        mlm_prob=mlm_prob,
    )

    # 接收dataloader参数
    dataloader_config = config_dict["dataloader"]
    batch_size = dataloader_config["batch_size"]
    shuffle = dataloader_config["shuffle"]
    num_workers = dataloader_config["num_workers"]
    pin_memory = dataloader_config["pin_memory"]
    drop_last = dataloader_config["drop_last"]
    # 实例化dataloader
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = drop_last,
        collate_fn = mlm_collator,
    )

    # 接收model参数
    model_config = config_dict["model"]
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
    train_metrics_config = metrics_config["train"]
    train_metric_list = []
    for train_metric in train_metrics_config:
        train_metric_type = train_metric["type"].strip().casefold()
        train_metric_params = train_metric["params"]
        train_metric_class = METRIC_REGISTRY[train_metric_type]
        train_metric_list.append(train_metric_class(**train_metric_params))
    train_metric_collection = MetricCollection(train_metric_list)
    valid_metrics_config = metrics_config["valid"]
    valid_metric_list = []
    for valid_metric in valid_metrics_config:
        valid_metric_type = valid_metric["type"].strip().casefold()
        valid_metric_params = valid_metric["params"]
        valid_metric_class = METRIC_REGISTRY[valid_metric_type]
        valid_metric_list.append(valid_metric_class(**valid_metric_params))
    valid_metric_collection = MetricCollection(valid_metric_list)
    test_metrics_config = metrics_config["test"]
    test_metric_list = []
    for test_metric in test_metrics_config:
        test_metric_type = test_metric["type"].strip().casefold()
        test_metric_params = test_metric["params"]
        test_metric_class = METRIC_REGISTRY[test_metric_type]
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
    train_fieldnames = base_fieldnames + loss_fieldnames + interval_loss_fieldnames + train_metric_fieldnames
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

    # 接收optimizer参数
    optimizer_config = config_dict["optimizer"]
    optimizer_type = optimizer_config["type"].strip().casefold()
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    betas = optimizer_config["betas"]
    eps = optimizer_config["eps"]
    betas = tuple(betas)
    # 获得optimizer的类
    optimizer_class = OPTIMIZER_REGISTRY[optimizer_type]
    # 实例化optimizer
    optimizer = optimizer_class(
        mlm_task.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    # 接收scheduler参数
    scheduler_config = config_dict["scheduler"]
    scheduler_type = scheduler_config["type"].strip().casefold()
    interval = scheduler_config["interval"]
    step_size = scheduler_config["step_size"]
    gamma = scheduler_config["gamma"]
    last_epoch = scheduler_config["last_epoch"]
    # 获得scheduler的类
    scheduler_class = SCHEDULER_REGISTRY[scheduler_type]
    # 实例化scheduler
    scheduler = scheduler_class(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch,
    )

    # 接收cuda参数
    cuda_config = config_dict["cuda"]
    cuda_device = cuda_config["device"]
    gpu_nums = cuda_config["gpu_nums"]

    # 接收train参数
    train_config = config_dict["train"]
    total_iters = train_config["total_iters"]
    log_every_iters = train_config["log_every_iters"]
    accumulation_steps = train_config["accumulation_steps"]
    do_validation = train_config["do_validation"]

    # 实例化trainer
    trainer = Trainer(
        optimizer=optimizer,
        scheduler=scheduler,
        device=cuda_device,
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
        if total_iters <= trainer.global_step:
            raise ValueError(f"resume_path: {resume}"
                             f"total_iters: {total_iters}"
                             f"trainer.global_step: {trainer.global_step}"
                             f"total_iters 必须大于 trainer.global_step 喵!")

    # do_validation 为True时接收valid参数，False时为None
    valid_dataloader = None
    valid_every_iters = None
    early_stop_patience = None
    if do_validation:
        valid_config = config_dict["valid"]
        valid_every_iters = valid_config["valid_every_iters"]
        early_stop_patience = valid_config["early_stop_patience"]
        validation_rounds_before_save_ckpt = valid_config["validation_rounds_before_save_ckpt"]
        valid_path = dataset_config["valid_path"]
        # 实例化dataset
        valid_dataset = MLMDataset(
            data_file_path=valid_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        # 实例化dataloader
        valid_dataloader = DataLoader(
            dataset = valid_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last = False,
            collate_fn = mlm_collator,
        )

    # test.enabled 为True时接收valid参数，False时为None
    test_config = config_dict["test"]
    test_enabled = test_config["enabled"]
    test_dataloader = None
    if test_enabled:
        test_path = dataset_config["test_path"]
        # 实例化dataset
        test_dataset = MLMDataset(
            data_file_path=test_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        # 实例化dataloader
        test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last = False,
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
            "total_iters": total_iters,
            "log_every_iters": log_every_iters,
            "accumulation_steps": accumulation_steps,
            "do_validation": do_validation,
            "valid_every_iters": valid_every_iters,
            "valid_dataloader": valid_dataloader,
            "early_stop_patience": early_stop_patience,
        }
    }

    if do_validation:
        checkpoints_dir = experiment_paths["checkpoints_dir"]
        train_params_dict["fit_kwargs"]["validation_rounds_before_save_ckpt"] = validation_rounds_before_save_ckpt
        train_params_dict["fit_kwargs"]["checkpoints_dir"] = checkpoints_dir


    return train_params_dict
