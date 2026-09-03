from registry.base import Registry
from parse.specs.component import EXAMPLE_TOKENIZER_SPEC, QA_TOKENIZER_SPEC, TinyBertForMaskedLM_SPEC, \
    TinyBertForQuestionAnswering_SPEC, CrossEntropyLoss_SPEC, AdamW_SPEC, StepLR, StepLR_SPEC, TokenAccuracy_SPEC, \
    ExactMatch_SPEC, CharacterF1_SPEC, TopKSamplingStrategy_SPEC

TOKENIZER_SPEC_CANONICAL_MAP = {
    'exampletokenizer':EXAMPLE_TOKENIZER_SPEC,
    'qatokenizer':QA_TOKENIZER_SPEC
}

TOKENIZER_SPEC_LOOKUP_MAP = {
    'exampletokenizer':'exampletokenizer',
    'qatokenizer':'qatokenizer',
    'example_tokenizer': 'exampletokenizer',
    'qa_tokenizer': 'qatokenizer'
}

TOKENIZER_SPEC_REGISTRY=Registry(
    category="tokenizer",
    canonical_map=TOKENIZER_SPEC_CANONICAL_MAP,
    lookup_map=TOKENIZER_SPEC_LOOKUP_MAP,
)

MODEL_SPEC_CANONICAL_MAP = {
    "tinybertformaskedlm": TinyBertForMaskedLM_SPEC,
    "tinybertforquestionanswering": TinyBertForQuestionAnswering_SPEC,
}

MODEL_SPEC_LOOKUP_MAP = {
    "tinybertformaskedlm": "tinybertformaskedlm",
    "tinybertforquestionanswering": "tinybertforquestionanswering",
    "tinybert_formaskedlm":"tinybertformaskedlm",
    "tinybert_forquestionanswering":"tinybertforquestionanswering",
    "tiny_bert_for_masked_lm": "tinybertformaskedlm",
    "tiny_bert_for_question_answering": "tinybertforquestionanswering",
}

MODEL_SPEC_REGISTRY = Registry(
    category="model",
    canonical_map=MODEL_SPEC_CANONICAL_MAP,
    lookup_map=MODEL_SPEC_LOOKUP_MAP,
)

LOSS_SPEC_CANONICAL_MAP = {
    "crossentropyloss": CrossEntropyLoss_SPEC,
}

LOSS_SPEC_LOOKUP_MAP = {
    "crossentropyloss": "crossentropyloss",
    "cross_entropy_loss": "crossentropyloss",
    "crossentropy_loss": "crossentropyloss",
}

LOSS_SPEC_REGISTRY = Registry(
    category="loss",
    canonical_map=LOSS_SPEC_CANONICAL_MAP,
    lookup_map=LOSS_SPEC_LOOKUP_MAP,
)

OPTIMIZER_SPEC_CANONICAL_MAP = {
    "adamw": AdamW_SPEC,
}

OPTIMIZER_SPEC_LOOKUP_MAP = {
    "adamw": "adamw",
    "adam_w": "adamw",
}

OPTIMIZER_SPEC_REGISTRY = Registry(
    category="optimizer",
    canonical_map=OPTIMIZER_SPEC_CANONICAL_MAP,
    lookup_map=OPTIMIZER_SPEC_LOOKUP_MAP,
)

SCHEDULER_SPEC_CANONICAL_MAP = {
    "steplr": StepLR_SPEC,
}

SCHEDULER_SPEC_LOOKUP_MAP = {
    "steplr": "steplr",
    "step_lr": "steplr",
}

SCHEDULER_SPEC_REGISTRY = Registry(
    category="scheduler",
    canonical_map=SCHEDULER_SPEC_CANONICAL_MAP,
    lookup_map=SCHEDULER_SPEC_LOOKUP_MAP,
)

METRICS_SPEC_CANONICAL_MAP = {
    "tokenaccuracy": TokenAccuracy_SPEC,
    "exactmatch": ExactMatch_SPEC,
    "characterf1": CharacterF1_SPEC,
}

METRICS_SPEC_LOOKUP_MAP = {
    "tokenaccuracy": "tokenaccuracy",
    "exactmatch": "exactmatch",
    "characterf1": "characterf1",
    "token_accuracy": "tokenaccuracy",
    "exact_match": "exactmatch",
    "character_f1": "characterf1",
}

METRICS_SPEC_REGISTRY = Registry(
    category="metric",
    canonical_map=METRICS_SPEC_CANONICAL_MAP,
    lookup_map=METRICS_SPEC_LOOKUP_MAP,
)

DECODING_SPEC_CANONICAL_MAP = {
    "topksampling": TopKSamplingStrategy_SPEC,
}

DECODING_SPEC_LOOKUP_MAP = {
    "topksampling": "topksampling",
    "topk_sampling": "topksampling",
    "top_k_sampling": "topksampling"
}

DECODING_SPEC_REGISTRY = Registry(
    category="decoding",
    canonical_map=DECODING_SPEC_CANONICAL_MAP,
    lookup_map=DECODING_SPEC_LOOKUP_MAP,
)