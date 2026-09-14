from .decoding import TopKSamplingStrategy, QASpanTopKDecodingStrategy

# DECODING_REGISTRY = {
#     "topksampling": TopKSamplingStrategy,
# }
#
# SUPPORTED_DECODING = set(DECODING_REGISTRY)

from registry.base import Registry

canonical_map = {
     "topksampling": TopKSamplingStrategy,
    "qaspantopkdecoding": QASpanTopKDecodingStrategy,
}

lookup_map = {
    "topksampling": "topksampling",
    "topk_sampling":"topksampling",
    "top_k_sampling": "topksampling",
    "qaspantopkdecoding": "qaspantopkdecoding",
    "qa_span_topk_decoding": "qaspantopkdecoding",
    "qa_span_top_k_decoding": "qaspantopkdecoding",
}

DECODING_REGISTRY = Registry(
    category='decoding',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_DECODING = set(DECODING_REGISTRY.keys())