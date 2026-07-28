SUPPORTED_MODEL_TYPE = {'transformer',}
from .Transformer.supported import (SUPPORTED_LOSSES,
                                   SUPPORTED_MODULES,
                                   SUPPORTED_TOKENIZERS,
                                   SUPPORTED_OPTIMIZERS,
                                   SUPPORTED_SCHEDULERS,
                                   SUPPORTED_EMBEDDINGS)

#不参与导入
SUPPORTED_MODELS = [SUPPORTED_MODEL_TYPE,
                    SUPPORTED_LOSSES,
                    SUPPORTED_MODULES,
                    SUPPORTED_TOKENIZERS,
                    SUPPORTED_OPTIMIZERS,
                    SUPPORTED_SCHEDULERS,
                    SUPPORTED_EMBEDDINGS]