from .losses import CrossEntropyLoss


# LOSS_REGISTRY = {
#     "crossentropyloss": CrossEntropyLoss,
# }
#
# SUPPORTED_LOSSES = set(LOSS_REGISTRY)

from registry.base import Registry

canonical_map = {
    "crossentropyloss": CrossEntropyLoss,
}

lookup_map = {
    "crossentropyloss": "crossentropyloss",
    "cross_entropy_loss": "crossentropyloss",
    # PARSE_REFACTOR_DEMO: YAML 示例使用 cross_entropy，该别名必须与
    # LOSS_REGISTRY 的 canonical key crossentropyloss 对齐。
    "cross_entropy": "crossentropyloss",
    "crossentropy_loss":"crossentropyloss",
}

LOSS_REGISTRY = Registry(
    category='loss',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_LOSSES = set(LOSS_REGISTRY.keys())