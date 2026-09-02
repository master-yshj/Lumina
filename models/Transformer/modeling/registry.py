from .modeling import TinyBertForMaskedLM, TinyBertForQuestionAnswering


# MODEL_REGISTRY = {
#     "tinybertformaskedlm": TinyBertForMaskedLM,
#     "tinybertforquestionanswering": TinyBertForQuestionAnswering,
# }
#
# SUPPORTED_MODEL_NAMES = set(MODEL_REGISTRY)

from registry.base import Registry

canonical_map = {
    "tinybertformaskedlm": TinyBertForMaskedLM,
    "tinybertforquestionanswering": TinyBertForQuestionAnswering,
}

lookup_map = {
    "tinybertformaskedlm":"tinybertformaskedlm",
    "tinybertforquestionanswering":"tinybertforquestionanswering",
    "tinybert_formaskedlm":"tinybertformaskedlm",
    "tinybert_forquestionanswering":"tinybertforquestionanswering",
    "tiny_bert_for_masked_lm": "tinybertformaskedlm",
    "tiny_bert_for_question_answering": "tinybertforquestionanswering",
}


MODEL_REGISTRY = Registry(
    category='model',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_MODEL_NAMES = set(MODEL_REGISTRY.keys())