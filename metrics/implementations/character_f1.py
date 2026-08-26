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
        if not type(self.total_f1) in (int, float) or self.total_f1 < 0 or not isfinite(self.total_f1):
            raise ValueError(f"total_f1: {self.total_f1} 必须为非负有限数值喵!")
        if not type(self.count) is int or self.count <= 0 or not isfinite(self.count):
            raise ValueError(f"count: {self.count} 必须为有限正整数喵!")
        summary_dict = {
            "character_f1": self.total_f1 / self.count,
        }
        return summary_dict

    def output_fields(self):
        return ["character_f1"]
