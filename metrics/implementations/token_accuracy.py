import torch

from metrics.core.base import Metric
from metrics.process.inputs import process_logits_labels

class TokenAccuracy(Metric):
    def __init__(self, ignore_index=-100):
        if not type(ignore_index) is int:
            raise ValueError(f"ignore_index必须为整型的喵! 现在的type: {type(ignore_index)}")
        super().__init__()
        self.ignore_index = ignore_index

        self.reset()

    def process(self, **context):
        process_dict = process_logits_labels(**context)
        return process_dict

    def update(self, **context):
        """
        :param context: batch: labels: [B,S], outputs: logits: [B,S,V]
        :return:
        """
        process_dict = self.process(**context)
        labels = process_dict["labels"]
        logits = process_dict["logits"]

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