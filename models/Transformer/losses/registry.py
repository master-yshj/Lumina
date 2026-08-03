from .losses import CrossEntropyLoss


LOSS_REGISTRY = {
    "crossentropyloss": CrossEntropyLoss,
}

SUPPORTED_LOSSES = set(LOSS_REGISTRY)