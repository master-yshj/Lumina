from metrics.core.base import Metric
from metrics.process.inputs import process_predictions_references

class ExactMatch(Metric):
    def __init__(self):
        super().__init__()
        self.exact_matched = 0
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
            for r_idx, reference in enumerate(reference_list):
                if prediction == reference:
                    self.exact_matched += 1
                    break
                else:
                    pass
            self.count += 1

    def compute(self):
        if not type(self.exact_matched) is int:
            raise ValueError(f"exact_matched 必须为整型的喵! 现在的type: {type(self.exact_matched)}")
        if not type(self.count) is int:
            raise ValueError(f"count 必须为整型的喵! 现在的type: {type(self.count)}")
        if not self.count >= 0:
            raise ValueError(f"count 必须为非负数喵! 现在是: {self.count}")
        if not self.exact_matched >= 0:
            raise ValueError(f"exact_matched 必须为非负数喵! 现在是: {self.exact_matched}")
        if not self.exact_matched <= self.count:
            raise ValueError(f"exact_matched 必须小于等于 count 喵!")
        if self.count == 0:
            raise ValueError(f"count 不能为0喵! 无法作为除数喵!")
        exact_match = self.exact_matched / self.count
        summary_dict = {
            "exact_match": exact_match,
        }
        return summary_dict

    def reset(self):
        self.exact_matched = 0
        self.count = 0

    def output_fields(self):
        return ["exact_match"]
