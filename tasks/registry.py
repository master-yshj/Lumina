from .MLM_PreTrain.builder import build_mlm_pretrain
from .MLM_PreTrain.test_builder import build_mlm_test
from .MLM_PreTrain.inference_builder import build_mlm_inference
# AI-CHANGE: MLM 可视化 Builder 接入(AI 编写, 待人工审查)。
# 只组装 tokenizer/model/dataloader/visualizers,不组装损失与指标。
from .MLM_PreTrain.visualize_builder import build_mlm_visualize
from .QA_SFT.builder import build_qa_sft
from .QA_SFT.test_builder import build_qa_test


# TASK_BUILDER_REGISTRY = {
#     "mlm_pretrain": build_mlm_pretrain,
#     "mlm_pretrain_test": build_mlm_test,
#     "mlm_pretrain_inference": build_mlm_inference,
#     "qa_sft": build_qa_sft,
#     "qa_sft_test": build_qa_test,
# }
#
# SUPPORTED_TASKS = set(TASK_BUILDER_REGISTRY)

from registry.base import Registry

canonical_map = {
    "mlmpretrain_train": build_mlm_pretrain,
    "mlmpretrain_test": build_mlm_test,
    "mlmpretrain_inference": build_mlm_inference,
    # AI-CHANGE: MLM 可视化任务 Builder(AI 编写, 待人工审查)。
    # 键按 train/test/inference 的后缀惯例取 mlmpretrain_visualize。
    "mlmpretrain_visualize": build_mlm_visualize,
    "qasft_train": build_qa_sft,
    "qasft_test": build_qa_test,
}

lookup_map = {
    "mlmpretrain_train": "mlmpretrain_train",
    "mlmpretrain_test": "mlmpretrain_test",
    "mlmpretrain_inference": "mlmpretrain_inference",
    # AI-CHANGE: MLM 可视化任务别名(AI 编写, 待人工审查)。
    "mlmpretrain_visualize": "mlmpretrain_visualize",
    "qasft_train": "qasft_train",
    "qasft_test": "qasft_test",
    "mlm_pretrain_train": "mlmpretrain_train",
    "mlm_pretrain_test": "mlmpretrain_test",
    "mlm_pretrain_inference": "mlmpretrain_inference",
    # AI-CHANGE: MLM 可视化任务别名(AI 编写, 待人工审查)。
    "mlm_pretrain_visualize": "mlmpretrain_visualize",
    "qa_sft_train": "qasft_train",
    "qa_sft_test": "qasft_test",
}

TASK_BUILDER_REGISTRY = Registry(
    category='task',
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_TASKS = set(TASK_BUILDER_REGISTRY.keys())