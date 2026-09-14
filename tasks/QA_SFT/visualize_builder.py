from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from models.Transformer.losses.registry import LOSS_REGISTRY
from models.Transformer.losses.losses import WeightedLoss
from tasks.QA_SFT.qa import QATask
from loaders.qa import QADataset, QACollator
from visualize.registry import VISUALIZER_REGISTRY
from Trainer.checkpoint import load_checkpoint

from torch.utils.data import DataLoader


def build_qa_visualize(config_dict, logger=None):
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"config_dict 必须为dict类型的喵! 现在的type: {type(config_dict)}"
        )
    if not config_dict:
        raise ValueError("config_dict 不能为空喵!")
    if "run" not in config_dict or "checkpoint" not in config_dict["run"]:
        raise ValueError("config_dict 必须包含 run.checkpoint 字段喵!")

    train_config_path = config_dict["run"]["checkpoint"]["train_config"]
    from parse.parser import parse_config

    train_config, _, _ = parse_config(
        config_path=train_config_path,
        expected_mode="train",
        validate_runtime=False,
    )

    tokenizer_config = train_config["tokenizer"]
    tokenizer_type = tokenizer_config["type"].strip().casefold()
    tokenizer_params = tokenizer_config["params"]
    tokenizer_key = TOKENIZER_REGISTRY.resolve_key(tokenizer_type)
    tokenizer_class = TOKENIZER_REGISTRY.get(tokenizer_key)
    tokenizer = tokenizer_class(
        vocab_file_path=tokenizer_params["vocab_file_path"],
        do_lower_case=tokenizer_params["do_lower_case"],
        strip_accents=tokenizer_params["strip_accents"],
        max_input_chars_per_word=tokenizer_params["max_input_chars_per_word"],
    )

    model_config = train_config["model"]
    model_type = model_config["type"].strip().casefold()
    model_params = model_config["params"]
    model_key = MODEL_REGISTRY.resolve_key(model_type)
    model_class = MODEL_REGISTRY.get(model_key)
    model = model_class(
        vocab_size=len(tokenizer),
        hidden_size=model_params["hidden_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
        type_vocab_size=model_params["type_vocab_size"],
        padding_idx=tokenizer.pad_token_id,
        num_heads=model_params["num_heads"],
        intermediate_size=model_params["intermediate_size"],
        num_layers=model_params["num_layers"],
        embedding_dropout=model_params["embedding_dropout"],
        attention_dropout=model_params["attention_dropout"],
        hidden_dropout=model_params["hidden_dropout"],
        layer_norm_eps=model_params["layer_norm_eps"],
    )

    qa_task = QATask(model=model, loss_fn=None)
    checkpoint = load_checkpoint(
        config_dict["run"]["checkpoint"]["path"],
        map_location="cpu",
    )
    if "task_state" not in checkpoint:
        raise ValueError("checkpoint 中必须包含 task_state 字段喵!")
    qa_task.load_state_dict(checkpoint["task_state"], strict=True)

    data_config = config_dict["data"]
    dataset_config = data_config["dataset"]
    visualize_path = dataset_config["splits"]["visualize"]
    visualize_dataset = QADataset(
        data_file_path=visualize_path,
        tokenizer=tokenizer,
        split_name="visualize",
        progress_interval=10000,
        read_interval=10000,
        logger=logger,
        max_position_embeddings=model_params["max_position_embeddings"],
    )
    qa_collator = QACollator(tokenizer=tokenizer)
    loader_config = data_config["loaders"]["visualize"]
    visualize_dataloader = DataLoader(
        dataset=visualize_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config["shuffle"],
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=loader_config["drop_last"],
        collate_fn=qa_collator,
    )

    visualizers = []
    visualizer_names = []
    for method_config in config_dict["visualize"]["methods"]:
        method_type = method_config["type"]
        method_key = VISUALIZER_REGISTRY.resolve_key(method_type)
        visualizer_class = VISUALIZER_REGISTRY.get(method_key)
        method_params = dict(method_config.get("params") or {})
        losses_config = method_params.pop("losses", None)
        if losses_config is not None:
            loss_list = []
            for loss_config in losses_config:
                loss_key = LOSS_REGISTRY.resolve_key(loss_config["type"])
                loss_class = LOSS_REGISTRY.get(loss_key)
                loss_list.append(
                    {
                        "name": loss_config["name"],
                        "loss_fn": loss_class(**(loss_config.get("params") or {})),
                        "weight": loss_config["weight"],
                    }
                )
            method_params["loss_fn"] = WeightedLoss(loss_list)
        visualizers.append(visualizer_class(**method_params))
        visualizer_names.append(method_key)

    if not visualizers:
        raise ValueError("visualizers 不能为空喵!")

    return {
        "task": qa_task,
        "model": model,
        "tokenizer": tokenizer,
        "visualize_dataloader": visualize_dataloader,
        "visualizers": visualizers,
        "visualizer_names": visualizer_names,
    }
