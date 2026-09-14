from copy import deepcopy
from models.Transformer.tokenizer.registry import TOKENIZER_REGISTRY
from models.Transformer.modeling.registry import MODEL_REGISTRY
from tasks.QA_SFT.qa import QATask
from tasks.QA_SFT.inference import QAInference
from Inference.decoding.registry import DECODING_REGISTRY

def build_qa_inference(config_dict):
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
    # 实例化task
    qa_task = QATask(
        model=model,
        loss_fn=None,
    )
    # 接收inference参数
    inference_dict = config_dict["inference"]
    doc_stride = inference_dict["doc_stride"]
    decoding_dict = inference_dict["decoding"]
    decoding_type = decoding_dict["type"].strip().casefold()
    decoding_params_dict = decoding_dict["params"]
    DECODING_CANONICAL_MAP = DECODING_REGISTRY.canonical_map
    decoding_class = DECODING_CANONICAL_MAP[decoding_type]
    decoding_strategy = decoding_class(**decoding_params_dict)
    # 构造QAInference
    qa_inference = QAInference(
        model=model,
        tokenizer=tokenizer,
        decoding_strategy=decoding_strategy,
        max_position_embeddings=max_position_embeddings,
        doc_stride=doc_stride
    )

    summary_dict = {
        "task": qa_task,
        "task_inference": qa_inference,
    }

    return summary_dict