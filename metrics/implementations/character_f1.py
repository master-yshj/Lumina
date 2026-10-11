from metrics.core.base import Metric
from collections import Counter
from math import isfinite
from metrics.process.inputs import process_predictions_references


class CharacterF1(Metric):
    def __init__(self):
        super().__init__()
        self.total_f1 = 0.0
        self.count = 0

    def process(self, **context):
        process_dict = process_predictions_references(**context)
        return process_dict

    def update(self, **context):
        process_dict = self.process(**context)
        predictions = process_dict["predictions"]
        references = process_dict["references"]
        for idx, prediction in enumerate(predictions):
            if not isinstance(prediction, str):
                raise ValueError(f"predictions[{idx}]: {prediction} 必须为字符串类型的喵! 现在的type: {type(prediction)}")
            reference_list = references[idx]
            if not isinstance(reference_list, list):
                raise ValueError(f"references[{idx}]: {reference_list} 必须为list类型的喵! 现在的type: {type(reference_list)}")
            if not reference_list:
                raise ValueError(f"references[{idx}] 不能为空喵! 即: predictions[{idx}]: {prediction} 至少要有一个reference喵!")
            for r_idx, reference in enumerate(reference_list):
                if not isinstance(reference, str):
                    raise ValueError(f"references[{idx}][{r_idx}]: {reference} 必须为字符串类型的喵! 现在的type: {type(reference)}")

        for idx, prediction in enumerate(predictions):
            reference_list = references[idx]
            f1_list = []
            for r_idx, reference in enumerate(reference_list):
                prediction_counter = Counter(prediction)
                reference_counter = Counter(reference)
                overlap = sum((prediction_counter & reference_counter).values())
                if prediction == "" and reference == "":
                    f1 = 1.0
                elif (prediction == "" and reference != "") or (reference == "" and prediction != ""):
                    f1 = 0.0
                else:
                    precision = overlap / len(prediction)
                    recall = overlap / len(reference)
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * precision * recall / (precision + recall)
                f1_list.append(f1)
            max_f1 = max(f1_list)
            self.total_f1 += max_f1
            self.count += 1

    def reset(self):
        self.total_f1 = 0.0
        self.count = 0

    def compute(self):
        state = self.distributed_state()
        result = self.compute_from_distributed_state(state)
        return result

    def output_fields(self):
        return ["character_f1"]

    def distributed_state(self):
        state = {"total_f1": self.total_f1, "count": self.count}
        return state

    def compute_from_distributed_state(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"state: {state} 必须为dict类型的喵! 现在的type: {type(state)}")
        if not set(state) == {'total_f1', 'count'}:
            missing_keys = {'total_f1', 'count'} - set(state)
            extra_keys = set(state) - {'total_f1', 'count'}
            if missing_keys:
                raise ValueError(f"state 中缺失了键: {missing_keys} 喵!")
            if extra_keys:
                raise ValueError(f"state 中多余了键: {extra_keys} 喵!")
        total_f1 = state['total_f1']
        count = state['count']
        if not type(total_f1) in (int, float) or not isfinite(total_f1) or total_f1 < 0:
            raise ValueError(f"total_f1: {total_f1} 必须为有限非负数值喵!")
        if not type(count) is int or count < 0:
            raise ValueError(f"count: {count} 必须为非负整数喵!")
        if count == 0:
            raise ValueError(f"count 不能为0喵! 无法作为除数喵!")
        score = total_f1 / count
        result = {"character_f1": score}
        return result