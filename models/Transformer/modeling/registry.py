from .modeling import TinyBertForMaskedLM


MODEL_REGISTRY = {
    "tinybertformaskedlm": TinyBertForMaskedLM,
}

SUPPORTED_MODEL_NAMES = set(MODEL_REGISTRY)