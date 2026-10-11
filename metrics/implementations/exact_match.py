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
        state = self.distributed_state()
        result = self.compute_from_distributed_state(state)
        return result

    def reset(self):
        self.exact_matched = 0
        self.count = 0

    def output_fields(self):
        return ["exact_match"]

    def distributed_state(self):
        state = {"exact_matched": self.exact_matched, "count": self.count}
        return state

    def compute_from_distributed_state(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"state: {state} 必须为dict类型的喵! 现在的type: {type(state)}")
        if not set(state) == {'exact_matched', 'count'}:
            missing_keys = {'exact_matched', 'count'} - set(state)
            extra_keys = set(state) - {'exact_matched', 'count'}
            if missing_keys:
                raise ValueError(f"state 中缺失了键: {missing_keys} 喵!")
            if extra_keys:
                raise ValueError(f"state 中多余了键: {extra_keys} 喵!")
        exact_matched = state['exact_matched']
        count = state['count']
        if not type(exact_matched) is int or exact_matched < 0:
            raise ValueError(f"exact_matched: {exact_matched} 必须为非负整数喵!")
        if not type(count) is int or count < 0:
            raise ValueError(f"count: {count} 必须为非负整数喵!")
        if not exact_matched <= count:
            raise ValueError(f"exact_matched: {exact_matched} 必须小于等于 count: {count} 喵!")
        if count == 0:
            raise ValueError(f"count 不能为0喵! 无法作为除数喵!")
        score = exact_matched / count
        result = {"exact_match": score}
        return result