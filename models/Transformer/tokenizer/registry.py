from .tokenizer import (
                    ExampleTokenizer,
                    QATokenizer)

TOKENIZER_REGISTRY = {
    'exampletokenizer':ExampleTokenizer,
    'qatokenizer':QATokenizer
}

SUPPORTED_TOKENIZERS = set(TOKENIZER_REGISTRY)

class TokenizerRegistry:
    """
    tokenizer的注册表
    """

    def register(self, canonical_key, alias, ):
