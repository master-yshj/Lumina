from .scheduler import StepLR

# SCHEDULER_REGISTRY = {
#     "steplr": StepLR,
# }
#
# SUPPORTED_SCHEDULERS = set(SCHEDULER_REGISTRY)

from registry.base import Registry

canonical_map = {
    "steplr": StepLR,
}

lookup_map = {
    "steplr": "steplr",
    "step_lr": "steplr",
}

SCHEDULER_REGISTRY = Registry(
    category='scheduler',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_SCHEDULERS = set(SCHEDULER_REGISTRY.keys())