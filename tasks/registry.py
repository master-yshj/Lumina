from .MLM_PreTrain.builder import build_mlm_pretrain
from .MLM_PreTrain.test_builder import build_mlm_test
from .MLM_PreTrain.inference_builder import build_mlm_inference
from .QA_SFT.builder import build_qa_sft
from .QA_SFT.test_builder import build_qa_test


TASK_BUILDER_REGISTRY = {
    "mlm_pretrain": build_mlm_pretrain,
    "mlm_pretrain_test": build_mlm_test,
    "mlm_pretrain_inference": build_mlm_inference,
    "qa_sft": build_qa_sft,
    "qa_sft_test": build_qa_test,
}

SUPPORTED_TASKS = set(TASK_BUILDER_REGISTRY)