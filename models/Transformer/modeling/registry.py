from .modeling import TinyBertForMaskedLM, TinyBertForQuestionAnswering


MODEL_REGISTRY = {
    "tinybertformaskedlm": TinyBertForMaskedLM,
    "tinybertforquestionanswering": TinyBertForQuestionAnswering,
}

SUPPORTED_MODEL_NAMES = set(MODEL_REGISTRY)