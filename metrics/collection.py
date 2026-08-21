from metrics.base import Metric

class MetricCollection(Metric):
    def __init__(self, metrics):
        if not isinstance(metrics, (tuple, list)):
            raise ValueError(f"metrics必须为tuple或者list类型的喵! 现在的type: {type(metrics)}")
        if not metrics:
            raise ValueError(f"metrics不能为空喵!")
        exist_keys = []
        extra_fields = []
        for element in metrics:
            if not isinstance(element, Metric):
                raise ValueError(f"metric 中的每个元素都必须为Metric实例喵! elem: {element} 的type: {type(element)}")
            output_fields = element.output_fields()
            if not isinstance(output_fields, (tuple, list)):
                raise ValueError(f"output_fields 必须为tuple或者list类型的喵! 现在的type: {type(output_fields)}")
            if not output_fields:
                raise ValueError(f"output_fields 不能为空喵!")
            for output_field in output_fields:
                if not isinstance(output_field, str):
                    raise ValueError(f"output_fields 中的元素必须为字符串类型的喵! 现在的type: {type(output_field)}")
                if not output_field.strip():
                    raise ValueError(f"output_fields 中的元素不能为空喵!")
                if not output_field in exist_keys:
                    exist_keys.append(output_field)
                else:
                    extra_fields.append(output_field)
        if extra_fields:
            raise ValueError(f"不同metric的output_fields不能重复喵! 重复的字段有: {extra_fields}")
        super().__init__()
        self.metrics = list(metrics)

    def update(self, **context):
        for metric in self.metrics:
            metric.update(**context)

    def compute(self):
        output_dict = {}
        for metric in self.metrics:
            metric_dict = metric.compute()
            if not isinstance(metric_dict, dict):
                raise ValueError(f"metric_dict 必须为dict类型的喵! 现在的type: {type(metric_dict)}")
            if not metric_dict:
                raise ValueError(f"metric_dict 不能为空喵!")
            declared_fields = set(metric.output_fields())
            actual_fields = set(metric_dict.keys())
            if not declared_fields == actual_fields:
                raise ValueError(f"metric: {metric} 实际计算返回的字段与预期不一致喵!")
            output_dict.update(metric_dict)
        return output_dict

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def output_fields(self):
        output_list = []
        for metric in self.metrics:
            output_list.extend(metric.output_fields())
        return output_list