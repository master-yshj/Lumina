from .MLM_PreTrain.builder import build_mlm_pretrain
from .MLM_PreTrain.test_builder import build_mlm_test


TASK_BUILDER_REGISTRY = {
    "mlm_pretrain": build_mlm_pretrain,
    "mlm_pretrain_test": build_mlm_test,
}

SUPPORTED_TASKS = set(TASK_BUILDER_REGISTRY)