from metrics.implementations.token_accuracy import TokenAccuracy
from metrics.implementations.exact_match import ExactMatch
from metrics.implementations.character_f1 import CharacterF1

METRIC_REGISTRY = {
    "tokenaccuracy": TokenAccuracy,
    "exact_match": ExactMatch,
    "character_f1": CharacterF1,
}

SUPPORTED_METRICS = set(METRIC_REGISTRY)