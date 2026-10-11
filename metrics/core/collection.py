import torch
import torch.distributed as dist

from math import isfinite

from metrics.core.base import Metric

class MetricCollection(Metric):
    def __init__(self, metrics):
        if not isinstance(metrics, (tuple, list)):
            raise ValueError(f"metrics必须为tuple或者list类型的喵! 现在的type: {type(metrics)}")
        # AI 编写：空 list/tuple 表示不计算附加指标；Loss 与早停独立于此集合。
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
        self._distributed_layout = None

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

    def export_distributed(self,):
        layouts = []
        states = []
        for metric in self.metrics:
            metric_class = type(metric)

            # AI 编写：若仍使用 Metric 的默认方法，说明没有提供统计导出实现。
            if metric_class.distributed_state is Metric.distributed_state:
                raise NotImplementedError(
                    f"指标 {metric_class.__name__} "
                    f"尚未实现 distributed_state 方法喵!"
                )

            # AI 编写：还必须能根据合并后的统计值计算最终指标。
            if (
                    metric_class.compute_from_distributed_state
                    is Metric.compute_from_distributed_state
            ):
                raise NotImplementedError(
                    f"指标 {metric_class.__name__} "
                    f"尚未实现 compute_from_distributed_state 方法喵!"
                )

            # AI 编写：调用当前指标的方法，返回值保存到局部变量 state。
            # 本项要求返回非空 dict，后面再检查其中的键和值。
            state = metric.distributed_state()

            if not isinstance(state, dict):
                raise TypeError(
                    f"指标 {metric_class.__name__} 的 distributed_state() "
                    f"必须返回 dict 喵! 现在的类型: {type(state)}"
                )

            if not state:
                raise ValueError(
                    f"指标 {metric_class.__name__} 的 distributed_state() "
                    f"不能返回空字典喵!"
                )

            # AI 编写：先检查当前指标 state 的全部键，确认可以按字符串排序。
            for state_key in state:
                if not isinstance(state_key, str):
                    raise TypeError(
                        f"指标 {metric_class.__name__} 的统计字段名必须为 str 喵! "
                        f"字段名: {state_key!r}，类型: {type(state_key)}"
                    )

                if not state_key.strip():
                    raise ValueError(
                        f"指标 {metric_class.__name__} 的统计字段名不能为空喵!"
                    )

            # AI 编写：每个指标单独创建 list[tuple[str, str]]，
            # 每项保存字段名称及数值类型，不保存该字段的当前数值。
            state_fields = []

            for state_key in sorted(state):
                value = state[state_key]

                if type(value) is int:
                    type_name = "int"
                elif type(value) is float:
                    if not isfinite(value):
                        raise ValueError(
                            f"指标 {metric_class.__name__} 的字段 "
                            f"{state_key!r} 必须为有限浮点数喵! 收到: {value!r}"
                        )
                    type_name = "float"
                else:
                    raise TypeError(
                        f"指标 {metric_class.__name__} 的字段 "
                        f"{state_key!r} 必须为严格的 Python int 或 float 喵! "
                        f"收到: {value!r}，类型: {type(value)}"
                    )

                state_fields.append((state_key, type_name))

                layout = {
                    "metric_type": (
                            metric_class.__module__ + "." + metric_class.__qualname__
                    ),
                    "output_fields": tuple(metric.output_fields()),
                    "state_fields": tuple(state_fields),
                }

                # AI 编写：两个列表按相同的指标顺序各追加一项，保持下标对应。
                layouts.append(layout)
                states.append(dict(state))

                # AI 编写：所有指标处理完后，返回由两个列表组成的二元组。
            return layouts, states

    def prepare_distributed(self, runtime_context):
        context = runtime_context
        if not context.enabled:
            return None
        exported = context.run_local('导出指标布局', self._export_distributed)
        layouts, states = exported
        gathered_layouts = [None] * context.world_size
        dist.all_gather_object(gathered_layouts, layouts)
        # AI 编写：reference 是 rank 0 导出的完整布局列表，作为比较基准。
        reference = gathered_layouts[0]

        # AI 编写：rank 是列表下标，对应进程编号；received 是该进程的布局列表。
        for rank, received in enumerate(gathered_layouts):
            if received != reference:
                raise ValueError(
                    f"跨 rank 的指标布局不一致喵!\n"
                    f"基准 rank: 0，布局: {reference!r}\n"
                    f"不一致的 rank: {rank}，布局: {received!r}"
                )

        # AI 编写：所有 rank 的布局一致后，保存本地已导出的布局列表。
        # [] 表示已经准备好的空指标集合，区别于尚未准备的 None。
        self._distributed_layout = layouts
        return None

    def compute_distributed(self, runtime_context):
        context = runtime_context
        if not context.enabled:
            output_dict = self.compute()
            return output_dict
        def collect_current():
            if self._distributed_layout is None:
                raise ValueError(f"prepare_distributed 尚未被调用喵! 请先调用该方法喵!")
            layouts, states = self.export_distributed()
            if not layouts == self._distributed_layout:
                raise ValueError(f"distributed_layout 出现变化了喵! 原layout: {self._distributed_layout!r} 现layout: {layouts!r}")
            return states

        local_states = context.run_local('检查本次指标状态', collect_current)
        if self._distributed_layout == []:
            return {}
        int_values = []
        float_values = []
        for metric_index, layout in enumerate(self._distributed_layout):
            for filed_name, type_name in layout["state_fields"]:
                value = local_states[filed_name]
                if type_name == "int":
                    int_values.append(value)
                elif type_name == "float":
                    float_values.append(value)
        reduce_device = context.device
        if context.backend == "gloo":
            reduce_device = torch.device("cpu")
        int_tensor = None
        float_tensor = None
        if int_values:
            int_tensor = torch.tensor(int_values, device=reduce_device, dtype=torch.int64)
        if float_values:
            float_tensor = torch.tensor(float_values, device=reduce_device, dtype=torch.float32)
        global_states = []
        int_index = 0
        float_index = 0
        for metric_index, layout in enumerate(self._distributed_layout):
            state = {}
            for filed_name, type_name in layout["state_fields"]:
                if type_name == "int":
                    state[filed_name] = int(int_tensor[int_index].item())
                    int_index += 1
                elif type_name == "float":
                    state[filed_name] = float(float_tensor[float_index].item())
                    float_index += 1
            global_states.append(state)
        def compute_global():
            # AI 编写：output_dict 是本次汇总的结果字典，在指标循环前创建。
            output_dict = {}

            for metric_index, metric in enumerate(self.metrics):
                # AI 编写：同一下标对应同一个指标及其统计字典。
                # 将统计字典作为参数传入，不写回 metric 的本地计数。
                metric_state = global_states[metric_index]
                metric_result = metric.compute_from_distributed_state(
                    metric_state
                )
                metric_name = type(metric).__name__

                # AI 编写：每项指标必须返回非空 dict。
                if not isinstance(metric_result, dict):
                    raise TypeError(
                        f"指标 {metric_name} 必须返回 dict 喵! "
                        f"现在的类型: {type(metric_result)}"
                    )

                if not metric_result:
                    raise ValueError(
                        f"指标 {metric_name} 不能返回空结果字典喵!"
                    )

                # AI 编写：按当前指标的下标，读取准备阶段保存的结果字段。
                expected_fields = set(
                    self._distributed_layout[metric_index]["output_fields"]
                )
                actual_fields = set(metric_result)

                if actual_fields != expected_fields:
                    missing_fields = expected_fields - actual_fields
                    extra_fields = actual_fields - expected_fields
                    raise ValueError(
                        f"指标 {metric_name} 的返回字段与声明不一致喵! "
                        f"缺失字段: {missing_fields!r}，"
                        f"多余字段: {extra_fields!r}"
                    )

                # AI 编写：合并之前检查重名，避免覆盖前一个指标的结果。
                duplicate_fields = set(output_dict) & actual_fields
                if duplicate_fields:
                    raise ValueError(
                        f"指标 {metric_name} 与前面的指标存在重复结果字段喵! "
                        f"重复字段: {duplicate_fields!r}"
                    )

                # AI 编写：全部检查通过后，将当前指标结果原地合并到 output_dict。
                output_dict.update(metric_result)

            # AI 编写：所有指标计算结束后，返回完整结果字典。
            return output_dict

        output_dict = context.run_local('计算全局指标', compute_global)
        return output_dict
