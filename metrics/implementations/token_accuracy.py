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
        state = self.distributed_state()
        result = self.compute_from_distributed_state(state)
        return result

    def output_fields(self):
        return ["token_accuracy"]

    def distributed_state(self):
        state = {'correct_count': self.correct_count, 'total_count': self.total_count}
        return state

    def compute_from_distributed_state(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"state: {state} 必须为dict类型的喵! 现在的type: {type(state)}")
        if not set(state) == {'correct_count', 'total_count'}:
            missing_keys = {'correct_count', 'total_count'} - set(state)
            extra_keys = set(state) - {'correct_count', 'total_count'}
            if missing_keys:
                raise ValueError(f"state 中缺失了键: {missing_keys} 喵!")
            if extra_keys:
                raise ValueError(f"state 中多余了键: {extra_keys} 喵!")
        correct_count = state['correct_count']
        total_count = state['total_count']
        if not type(correct_count) is int or correct_count < 0:
            raise ValueError(f"correct_count: {correct_count} 必须为非负整数喵!")
        if not type(total_count) is int or total_count < 0:
            raise ValueError(f"total_count: {total_count} 必须为非负整数喵!")
        if not correct_count <= total_count:
            raise ValueError(f"correct_count: {correct_count} 必须小于等于 total_count: {total_count} 喵!")
        if total_count == 0:
            raise ValueError(f"total_count 不能为0喵! 无法作为除数喵!")
        score = correct_count / total_count
        result = {"token_accuracy": score}
        return result