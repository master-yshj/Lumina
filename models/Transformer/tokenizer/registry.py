from .tokenizer import (
                    ExampleTokenizer,
                    QATokenizer)

# TOKENIZER_REGISTRY = {
#     'exampletokenizer':ExampleTokenizer,
#     'qatokenizer':QATokenizer
# }
#
# SUPPORTED_TOKENIZERS = set(TOKENIZER_REGISTRY)

from registry.base import Registry

canonical_map = {
    'exampletokenizer':ExampleTokenizer,
    'qatokenizer':QATokenizer
}

lookup_map = {
    'exampletokenizer':'exampletokenizer',
    'qatokenizer':'qatokenizer',
    'example_tokenizer': 'exampletokenizer',
    'qa_tokenizer': 'qatokenizer'
}

TOKENIZER_REGISTRY = Registry(
    category='tokenizer',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_TOKENIZERS = set(TOKENIZER_REGISTRY.keys())