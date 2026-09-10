from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from loaders.qa import QADataset, QACollator
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from tasks.QA_SFT.qa import QATask
from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
from metrics.registry import METRIC_REGISTRY
from metrics.core.collection import MetricCollection
from metrics.output.writer import MetricWriter
from Trainer.trainer import Trainer
# AI-CHANGE: 为 init_from backbone 加载补充的导入（原加载逻辑依赖
# Trainer.checkpoint.load_checkpoint，迁移时该导入随加载动作一起丢失）。
from Trainer.checkpoint import load_checkpoint, load_checkpoint_state

from torch.utils.data import DataLoader

def build_qa_sft(config_dict, device_summary, resume=None, experiment_paths=None, logger=None):
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

    # 接收dataset参数
    data_config = config_dict["data"]
    dataset_config = data_config["dataset"]
    dataset_splits_config = dataset_config["splits"]
    dataset_params_config = dataset_config["params"]
    train_path = dataset_splits_config["train"]
    # 实例化训练集
    train_dataset = QADataset(
        data_file_path=train_path,
        tokenizer=tokenizer,
        split_name="train",
        progress_interval=10000,
        read_interval=10000,
        logger=logger,
        max_position_embeddings=max_position_embeddings
    )
    # 实例化collator
    qa_collator = QACollator(
        tokenizer=tokenizer,
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
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        num_workers=train_num_workers,
        pin_memory=train_pin_memory,
        drop_last=train_drop_last,
        collate_fn=qa_collator,
    )

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

    # ========================= AI 修改开始 =========================
    # AI-CHANGE: 恢复旧版 backbone 加载逻辑（来源: git commit 46be1fa 的
    # tasks/QA_SFT/builder.py 第 189-199 行，逐行照抄旧版实现）。
    #
    # 背景: 配置结构从 model.backbone_path 迁移为 model.init_from 时，
    # Builder 里丢失了加载动作，只留下 model_init_from = ... 这行死代码，
    # 导致 QA 模型始终从随机权重训练，这是 valid 指标为 0 的重要代码级原因。
    #
    # 加载方式与旧版完全一致：
    #   1. 复用现成的 load_checkpoint() 读取并校验 .pt 文件；
    #   2. 取出 checkpoint["task_state"]；
    #   3. 只保留 "model.bert." 前缀的键（backbone 部分）；
    #   4. 剥离前缀后加载进 model.bert 子模块；
    #   5. strict=True 只作用于过滤后的 backbone 子集，MLM head 等
    #      无关键不会冲突，同时保证 backbone 必须完整匹配，不静默漏加载。
    #
    # AI 新增的防御（旧版没有，仅提前报错并说明原因，行为不变）:
    #   - kind 非 "backbone" 时报错；
    #   - 过滤后 backbone_state 为空时报错（否则 strict=True 会以
    #     missing keys 的形式报错，原因不直观）。
    #
    # 预期日志验证点: 启动训练不报错即说明加载成功；MLM 的 mlm_head.* 和
    # QA 的 qa_outputs.* 权重保持随机初始化，属于预期行为。
    if model_init_from is not None:
        if model_init_from["kind"] != "backbone":
            raise ValueError(
                f"init_from.kind: {model_init_from['kind']} 当前仅支持 backbone 喵!"
            )
        backbone_checkpoint = load_checkpoint(model_init_from["path"])
        task_state = backbone_checkpoint["task_state"]
        backbone_prefix = "model.bert."
        backbone_state = {
            key[len(backbone_prefix):]: value
            for key, value in task_state.items()
            if key.startswith(backbone_prefix)
        }
        if not backbone_state:
            raise ValueError(
                f"checkpoint 中没有找到任何 {backbone_prefix} 前缀的 backbone 权重喵! "
                f"path: {model_init_from['path']}"
            )
        model.bert.load_state_dict(backbone_state, strict=True)
    # ========================= AI 修改结束 =========================

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
    qa_task = QATask(
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
        qa_task.parameters(),
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
        task=qa_task,
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
            task=qa_task,
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
        valid_dataset = QADataset(
            data_file_path=valid_path,
            tokenizer=tokenizer,
            split_name="valid",
            progress_interval=10000,
            read_interval=10000,
            logger=logger,
            max_position_embeddings=max_position_embeddings,
        )
        # 实例化dataloader
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=valid_batch_size,
            shuffle=valid_shuffle,
            num_workers=valid_num_workers,
            pin_memory=valid_pin_memory,
            drop_last=valid_drop_last,
            collate_fn=qa_collator,
        )

    # test.enabled 为True时接收valid参数，False时为None
    test_dataloader = None
    if test_after_train:
        test_path = dataset_splits_config["test"]
        # 实例化dataset
        test_dataset = QADataset(
            data_file_path=test_path,
            tokenizer=tokenizer,
            split_name="test",
            progress_interval=10000,
            read_interval=10000,
            logger=logger,
            max_position_embeddings=max_position_embeddings,
        )
        # 实例化dataloader
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            shuffle=test_shuffle,
            num_workers=test_num_workers,
            pin_memory=test_pin_memory,
            drop_last=test_drop_last,
            collate_fn=qa_collator,
        )

    # 组装参数字典
    train_params_dict = {
        "trainer": trainer,
        "task": qa_task,
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