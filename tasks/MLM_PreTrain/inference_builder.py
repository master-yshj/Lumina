from copy import deepcopy
from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from tasks.MLM_PreTrain.mlm import MLMTask
from tasks.MLM_PreTrain.inference import MLMInference
from Inference.decoding.registry import DECODING_REGISTRY

def build_mlm_inference(config_dict):
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
    # 实例化task
    mlm_task = MLMTask(
        model=model,
        loss_fn=None,
    )
    # 接收inference参数
    inference_dict = config_dict["inference"]
    decoding_dict = inference_dict["decoding"]
    decoding_name = decoding_dict["name"].strip().casefold()
    copy_decoding_dict = deepcopy(decoding_dict)
    del copy_decoding_dict["name"]
    strategy_params = copy_decoding_dict
    decoding_class = DECODING_REGISTRY[decoding_name]
    decoding_strategy = decoding_class(**strategy_params)
    max_seq_length = inference_dict["max_seq_length"]
    # 构造MLMInference
    mlm_inference = MLMInference(
        model=model,
        tokenizer=tokenizer,
        decoding_strategy=decoding_strategy,
        max_seq_length=max_seq_length,
    )

    summary_dict = {
        "task": mlm_task,
        "task_inference": mlm_inference,
    }

    return summary_dict