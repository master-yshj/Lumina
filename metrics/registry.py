from .accuracy import TokenAccuracy

METRIC_REGISTRY = {
    "tokenaccuracy": TokenAccuracy,
}

SUPPORTED_METRICS = set(METRIC_REGISTRY)