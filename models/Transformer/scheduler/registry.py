from .scheduler import StepLR

SCHEDULER_REGISTRY = {
    "steplr": StepLR,
}

SUPPORTED_SCHEDULERS = set(SCHEDULER_REGISTRY)