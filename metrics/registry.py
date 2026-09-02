from metrics.implementations.token_accuracy import TokenAccuracy
from metrics.implementations.exact_match import ExactMatch
from metrics.implementations.character_f1 import CharacterF1

# METRIC_REGISTRY = {
#     "tokenaccuracy": TokenAccuracy,
#     "exact_match": ExactMatch,
#     "character_f1": CharacterF1,
# }
#
# SUPPORTED_METRICS = set(METRIC_REGISTRY)

from registry.base import Registry

canonical_map = {
     "tokenaccuracy": TokenAccuracy,
     "exactmatch": ExactMatch,
     "characterf1": CharacterF1,
}

lookup_map = {
    "tokenaccuracy": "tokenaccuracy",
    "exactmatch": "exactmatch",
    "characterf1": "characterf1",
    "token_accuracy": "tokenaccuracy",
    "exact_match": "exactmatch",
    "character_f1": "characterf1",
}

METRIC_REGISTRY = Registry(
    category='metric',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_METRICS = set(METRIC_REGISTRY.keys())