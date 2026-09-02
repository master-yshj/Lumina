from .optimizer import AdamW

# OPTIMIZER_REGISTRY = {
#     "adamw": AdamW,
# }
#
# SUPPORTED_OPTIMIZERS = set(OPTIMIZER_REGISTRY)

from registry.base import Registry

canonical_map = {
    "adamw": AdamW,
}

lookup_map = {
    "adamw": "adamw",
    "adam_w": "adamw",
}

OPTIMIZER_REGISTRY = Registry(
    category='optimizer',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_OPTIMIZERS = set(OPTIMIZER_REGISTRY.keys())