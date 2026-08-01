SUPPORTED_MODEL_TYPES = {'transformer',}
from .Transformer.supported import (SUPPORTED_LOSSES,
                                   SUPPORTED_TOKENIZERS,
                                   SUPPORTED_OPTIMIZERS,
                                   SUPPORTED_SCHEDULERS,
                                   SUPPORTED_MODEL_NAMES)

#不参与导入
SUPPORTED_MODELS = [SUPPORTED_MODEL_TYPES,
                    SUPPORTED_LOSSES,
                    SUPPORTED_TOKENIZERS,
                    SUPPORTED_OPTIMIZERS,
                    SUPPORTED_SCHEDULERS,
                    SUPPORTED_MODEL_NAMES]