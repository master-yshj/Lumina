from .optimizer import AdamW

OPTIMIZER_REGISTRY = {
    "adamw": AdamW,
}

SUPPORTED_OPTIMIZERS = set(OPTIMIZER_REGISTRY)