from .decoding import TopKSamplingStrategy

# DECODING_REGISTRY = {
#     "topksampling": TopKSamplingStrategy,
# }
#
# SUPPORTED_DECODING = set(DECODING_REGISTRY)

from registry.base import Registry

canonical_map = {
     "topksampling": TopKSamplingStrategy,
}

lookup_map = {
    "topksampling": "topksampling",
    "topk_sampling":"topksampling",
    "top_k_sampling": "topksampling"
}

DECODING_REGISTRY = Registry(
    category='decoding',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_DECODING = set(DECODING_REGISTRY.keys())