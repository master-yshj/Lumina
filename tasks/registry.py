from .MLM_PreTrain.builder import build_mlm_pretrain


TASK_BUILDER_REGISTRY = {
    "mlm_pretrain": build_mlm_pretrain,
}

SUPPORTED_TASKS = set(TASK_BUILDER_REGISTRY)