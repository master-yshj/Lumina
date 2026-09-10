from .scheduler import StepLR, WarmupScheduler

# SCHEDULER_REGISTRY = {
#     "steplr": StepLR,
#     "warmup": WarmupScheduler,
# }
#
# SUPPORTED_SCHEDULERS = set(SCHEDULER_REGISTRY)

from registry.base import Registry

canonical_map = {
    "steplr": StepLR,
    "warmup": WarmupScheduler,
}

lookup_map = {
    "steplr": "steplr",
    "step_lr": "steplr",
    "warmup": "warmup",
    "warmup_scheduler": "warmup",
}

SCHEDULER_REGISTRY = Registry(
    category='scheduler',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_SCHEDULERS = set(SCHEDULER_REGISTRY.keys())