from .decoding import TopKSamplingStrategy

DECODING_REGISTRY = {
    "top_k_sampling": TopKSamplingStrategy,
}

SUPPORTED_DECODING = set(DECODING_REGISTRY)