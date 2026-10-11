import torch
import torch.distributed as dist

from math import isfinite

class LossAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.batch_count = 0
        self.loss_sums = {}
        return None

    def update(self, step_result):
        if not isinstance(step_result, dict):
            raise TypeError(f"step_result: {step_result} 必须为dict类型的喵! 现在的type: {type(step_result)}")
        if not "loss" in step_result:
            raise ValueError(f"step_result: {step_result} 不能缺少loss字段喵!")
        if not "losses" in step_result:
            raise ValueError(f"step_result: {step_result} 不能缺少losses字段喵!")
        loss = step_result["loss"]
        losses = step_result["losses"]
        if not type(loss) in (int, float) or not isfinite(loss):
            raise ValueError(f"loss: {loss} 必须为有限整型或浮点型数值喵!")
        if not isinstance(losses, dict):
            raise TypeError(f"losses: {losses} 必须为dict类型的喵! 现在的type: {type(losses)}")
        checked_losses = {}
        for loss_name, loss_detail in losses.items():
            if not isinstance(loss_name, str):
                raise TypeError(f"loss_name: {loss_name} 必须为str类型的喵! 现在的type: {type(loss_name)}")
            if not loss_name.strip():
                raise ValueError(f"loss_name 不能为空喵!")
            if not isinstance(loss_detail, dict):
                raise TypeError(f"loss_detail: {loss_detail} 必须为dict类型的喵! 现在的type: {type(loss_detail)}")
            if not loss_detail:
                raise ValueError(f"loss_detail 不能为空喵!")
            if not "raw" in loss_detail:
                raise ValueError(f"loss_detail: {loss_detail} 必须含有raw字段喵!")
            if not "weight" in loss_detail:
                raise ValueError(f"loss_detail: {loss_detail} 必须含有weight字段喵!")
            if not "weighted" in loss_detail:
                raise ValueError(f"loss_detail: {loss_detail} 必须含有weighted字段喵!")
            loss_raw = loss_detail["raw"]
            loss_weight = loss_detail["weight"]
            loss_weighted = loss_detail["weighted"]
            if not type(loss_raw) in (int, float) or not isfinite(loss_raw):
                raise ValueError(f"loss_raw: {loss_raw} 必须为有限数值喵!")
            if not type(loss_weight) in (int, float) or not isfinite(loss_weight):
                raise ValueError(f"loss_weight: {loss_weight} 必须为有限数值喵!")
            if not type(loss_weighted) in (int, float) or not isfinite(loss_weighted):
                raise ValueError(f"loss_weighted: {loss_weighted} 必须为有限数值喵!")
            checked_losses[loss_name] = {
                "raw": loss_raw,
                "weight": loss_weight,
                "weighted": loss_weighted,
            }
        if self.batch_count > 0:
            current_names = set(checked_losses)
            recorded_names = set(self.loss_sums)

            if current_names != recorded_names:
                missing_names = recorded_names - current_names
                extra_names = current_names - recorded_names
                raise ValueError(
                    f"本批损失名称与已有累计记录不一致喵! "
                    f"缺失名称: {sorted(missing_names)}，"
                    f"新增名称: {sorted(extra_names)}"
                )

            for loss_name, loss_detail in checked_losses.items():
                current_weight = loss_detail["weight"]
                recorded_weight = self.loss_sums[loss_name]["weight"]

                if current_weight != recorded_weight:
                    raise ValueError(
                        f"损失 {loss_name!r} 的权重发生变化喵! "
                        f"已有权重: {recorded_weight}，"
                        f"本批权重: {current_weight}"
                    )
        elif self.batch_count == 0:
            for loss_name, loss_detail in checked_losses.items():
                self.loss_sums[loss_name] = {
                    "raw_sum": 0.0,
                    "weighted_sum": 0.0,
                    "weight": loss_detail["weight"]
                }
        self.loss_sum += loss
        self.batch_count += 1
        for loss_name, loss_detail in checked_losses.items():
            record = self.loss_sums[loss_name]
            record["raw_sum"] += loss_detail["raw"]
            record["weighted_sum"] += loss_detail["weighted"]

    def _make_summary(self, total, count, records):
        if count == 0:
            return {'avg_loss':None,'losses':{},'completed_batches':0}
        else:
            loss_details = {}
            for loss_name, loss_detail in records.items():
                loss_details[loss_name] = {
                    "avg_raw": loss_detail["raw_sum"] / count,
                    "avg_weighted": loss_detail["weighted_sum"] / count,
                    "weight": loss_detail["weight"],
                }
        return {'avg_loss':total/count,'losses':loss_details,'completed_batches':count}

    def summarize(self,runtime_context=None):
        if runtime_context is None or runtime_context.enabled is False:
            return self._make_summary(self.loss_sum, self.batch_count, self.loss_sums)
        else:
            context = runtime_context
            def collect_local():
                if not type(self.batch_count) is int or self.batch_count < 0:
                    raise ValueError(f"batch_count: {self.batch_count} 必须为非负整数喵!")
                if not isfinite(self.loss_sum):
                    raise ValueError(f"loss_sum: {self.loss_sum} 必须为有限值喵!")
                for loss_name, loss_detail in self.loss_sums.items():
                    raw_sum = loss_detail["raw_sum"]
                    weighted_sum = loss_detail["weighted_sum"]
                    weight = loss_detail["weight"]
                    if not isfinite(raw_sum):
                        raise ValueError(f"loss_name: {loss_name} raw_sum: {raw_sum} 必须为有限值喵!")
                    if not isfinite(weighted_sum):
                        raise ValueError(f"loss_name: {loss_name} weighted_sum: {weighted_sum} 必须为有限值喵!")
                    if not isfinite(weight):
                        raise ValueError(f"loss_name: {loss_name} weight: {weight} 必须为有限值喵!")
                if self.batch_count == 0:
                    if self.loss_sums:
                        raise ValueError(f"batch_count 为 0 时, loss_sums 必须为None喵!")
                layout = tuple((name, self.loss_sums[name]['weight']) for name in sorted(self.loss_sums))
                return {'count': self.batch_count, 'layout': layout}
        description = context.run_local(
            "检查 Loss 累计状态",
            collect_local
        )

        descriptions = [None] * context.world_size
        dist.all_gather_object(descriptions, description)

        nonempty = []
        for rank, gathered_description in enumerate(descriptions):
            if gathered_description["count"] > 0:
                nonempty.append((rank, gathered_description))

        if not nonempty:
            return self._make_summary(
                self.loss_sum,
                self.batch_count,
                self.loss_sums
            )

        base_rank, base_description = nonempty[0]
        layout = base_description["layout"]

        for rank, gathered_description in nonempty[1:]:
            current_layout = gathered_description["layout"]

            if current_layout != layout:
                base_weights = dict(layout)
                current_weights = dict(current_layout)

                differences = []
                all_names = sorted(
                    set(base_weights) | set(current_weights)
                )

                for loss_name in all_names:
                    if loss_name not in base_weights:
                        differences.append(
                            f"{loss_name!r} 在基准 rank 中不存在，"
                            f"当前 rank 权重为 {current_weights[loss_name]!r}"
                        )
                    elif loss_name not in current_weights:
                        differences.append(
                            f"{loss_name!r} 在当前 rank 中不存在，"
                            f"基准 rank 权重为 {base_weights[loss_name]!r}"
                        )
                    elif base_weights[loss_name] != current_weights[loss_name]:
                        differences.append(
                            f"{loss_name!r} 的权重不一致："
                            f"基准 rank={base_rank} 的权重为 "
                            f"{base_weights[loss_name]!r}，"
                            f"当前 rank={rank} 的权重为 "
                            f"{current_weights[loss_name]!r}"
                        )

                raise ValueError(
                    f"跨 rank 的 Loss 布局不一致喵！"
                    f"当前 rank: {rank}，"
                    f"差异: {differences}"
                )

        values = [self.loss_sum]

        for loss_name, loss_weight in layout:
            if self.batch_count > 0:
                record = self.loss_sums[loss_name]
                values.extend([
                    record["raw_sum"],
                    record["weighted_sum"],
                ])
            else:
                values.extend([0.0, 0.0])

        reduce_device = context.device
        if context.backend == "gloo":
            reduce_device = torch.device("cpu")
        sums_tensor = torch.tensor(values, dtype=torch.float64, device=reduce_device)
        count_tensor = torch.tensor([self.batch_count], dtype=torch.int64, device=reduce_device)
        dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        total = float(sums_tensor[0].item())
        count = int(count_tensor[0].item())
        records = {}
        for i, (name, weight) in enumerate(layout):
            raw_sum = sums_tensor[1+2*i].item()
            weighted_sum = sums_tensor[2+2*i].item()
            records[name] = {
                "raw_sum": raw_sum,
                "weighted_sum": weighted_sum,
                "weight": weight,
            }
        return self._make_summary(total, count, records)