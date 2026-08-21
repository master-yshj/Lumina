import torch

from metrics.base import Metric

class TokenAccuracy(Metric):
    def __init__(self, ignore_index=-100):
        if not type(ignore_index) is int:
            raise ValueError(f"ignore_index必须为整型的喵! 现在的type: {type(ignore_index)}")
        super().__init__()
        self.ignore_index = ignore_index

        self.reset()

    def update(self, **context):
        """
        :param context: batch: labels: [B,S], outputs: logits: [B,S,V]
        :return:
        """
        if not "batch" in context:
            raise ValueError(f"context中不包含batch这个键喵!")
        if not "outputs" in context:
            raise ValueError(f"context中不包含outputs这个键喵!")
        batch = context["batch"]
        outputs = context["outputs"]
        if not isinstance(batch, dict):
            raise ValueError(f"context.batch 必须为字典类型的喵! 现在的type: {type(batch)}")
        if not batch:
            raise ValueError(f"context.batch 不能为空喵!")
        if not "labels" in batch:
            raise ValueError(f"context.batch 中必须包含名为labels的键喵!")
        if not isinstance(outputs, dict):
            raise ValueError(f"context.outputs 必须为字典类型的喵! 现在的type: {type(outputs)}")
        if not outputs:
            raise ValueError(f"context.outputs 不能为空喵!")
        if not "logits" in outputs:
            raise ValueError(f"context.outputs 中必须包含名为logits的键喵!")
        labels = batch["labels"]
        logits = outputs["logits"]
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"context.batch.labels 必须为Tensor类型的喵! 现在的type: {type(labels)}")
        if not labels.dim() == 2:
            raise ValueError(f"context.batch.labels 必须为2维的喵! 现在的维度: {labels.dim()}")
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"context.outputs.logits 必须为Tensor类型的喵! 现在的type: {type(logits)}")
        if not logits.dim() == 3:
            raise ValueError(f"context.outputs.logits 必须为3维的喵! 现在的维度: {logits.dim()}")

        if not logits.shape[:2] == labels.shape:
            raise ValueError(f"labels的前两维必须和logits保持一致喵!")
        if not logits.size(-1) > 0:
            raise ValueError(f"logits的最后一维必须大于0喵!")

        if not logits.is_floating_point():
            raise ValueError(f"logits必须为浮点类型的Tensor喵! 现在的dtype: {logits.dtype}")
        if not labels.dtype == torch.long:
            raise ValueError(f"labels必须为long类型的Tensor喵! 现在的dtype: {labels.dtype}")

        if not logits.device == labels.device:
            raise ValueError(f"logits和labels的设备必须保持一致喵! logits的device: {logits.device} labels的device: {labels.device}")

        valid_mask = labels != self.ignore_index
        predictions_pos = logits.argmax(dim=-1)

        valid_count = valid_mask.sum().item()
        correct_count = ((predictions_pos == labels) & valid_mask).sum().item()

        self.correct_count += correct_count
        self.total_count += valid_count

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def compute(self):
        if not type(self.correct_count) is int:
            raise ValueError(f"correct_count 必须为整型的喵! 现在的type: {type(self.correct_count)}")
        if not type(self.total_count) is int:
            raise ValueError(f"total_count 必须为整型的喵! 现在的type: {type(self.total_count)}")
        if not self.total_count >= 0:
            raise ValueError(f"total_count 必须为非负数喵! 现在是: {self.total_count}")
        if not self.correct_count >= 0:
            raise ValueError(f"correct_count 必须为非负数喵! 现在是: {self.correct_count}")
        if not self.correct_count <= self.total_count:
            raise ValueError(f"correct_count 必须小于等于 total_count 喵!")
        if self.total_count == 0:
            raise ValueError(f"total_count 不能为0喵! 无法作为除数喵!")
        token_accuracy = self.correct_count / self.total_count
        summary_dict = {
            "token_accuracy": token_accuracy,
        }
        return summary_dict

    def output_fields(self):
        return ["token_accuracy"]