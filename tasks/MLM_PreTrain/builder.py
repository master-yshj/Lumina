from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from loaders.mlm import MLMDataset, MLMCollator
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from tasks.MLM_PreTrain.mlm import MLMTask
from models.Transformer.optimizer.registry import OPTIMIZER_REGISTRY
from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
from Trainer.trainer import Trainer

from torch.utils.data import DataLoader

def build_mlm_pretrain(config_dict, resume=None):
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

    # 实例化trainer
    trainer = Trainer(
        optimizer=optimizer,
        scheduler=scheduler,
        device=cuda_device,
        task=mlm_task,
    )

    # 接收train参数
    train_config = config_dict["train"]
    total_iters = train_config["total_iters"]
    log_every_iters = train_config["log_every_iters"]

    # 组装参数字典
    train_params_dict = {}
    train_params_dict = {
        "trainer": trainer,
        "train_dataloader": train_dataloader,
        "fit_kwargs": {
            "total_iters": total_iters,
            "log_every_iters": log_every_iters,
        }
    }

    return train_params_dict
