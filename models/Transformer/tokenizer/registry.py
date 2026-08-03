from .tokenizer import (
                    ExampleTokenizer,
                    QATokenizer)

TOKENIZER_REGISTRY = {
    'exampletokenizer':ExampleTokenizer,
    'qatokenizer':QATokenizer
}

SUPPORTED_TOKENIZERS = set(TOKENIZER_REGISTRY)