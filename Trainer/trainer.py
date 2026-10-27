import torch
import torch.distributed as dist
import pathlib

from math import isfinite
from contextlib import nullcontext

from Trainer.checkpoint import build_checkpoint, save_checkpoint, _validate_training_progress
from device import get_memory_summary, reset_peak_memory_stats, get_memory_warning, get_oom_info
from runtime.distributed import RuntimeContext
from Trainer.precision import PrecisionController
from Trainer.statistics import LossAccumulator
from metrics.core.collection import MetricCollection
from torch.utils.data.distributed import DistributedSampler

class Trainer:
    def __init__(self, task, optimizer, device, device_summary, train_metric_collection, valid_metric_collection, train_metrics_writer, valid_metrics_writer, scheduler=None, logger=None, *, runtime_context=None, precision=None):
        self.task = task
        self.train_task = self.task
        self.optimizer = optimizer
        self.device = torch.device(device)
        if runtime_context is None:
            runtime_context = RuntimeContext(device=self.device)
        elif not isinstance(runtime_context, RuntimeContext):
            raise TypeError(f"runtime_context 不为None时, 必须为RuntimeContext类型的喵! 现在的type: {type(runtime_context)}")
        if precision is None:
            precision = PrecisionController(device=self.device, precision="fp32")
        elif not isinstance(precision, PrecisionController):
            raise TypeError(f"precision 不为None时, 必须为PrecisionController类型的喵! 现在的type: {type(precision)}")
        if not self.device == runtime_context.device == precision.device:
            raise ValueError(f"device, runtime_context的设备, precision的设备三者不一致喵!\n"
                             f"device: {self.device}\n"
                             f"runtime_context.device: {runtime_context.device}\n"
                             f"precision.device: {precision.device}\n")
        self.runtime_context = runtime_context
        self.precision = precision
        self.device_summary = device_summary
        self.scheduler = scheduler
        self.logger = logger

        if not isinstance(train_metric_collection, MetricCollection):
            raise TypeError(f"train_metric_collection: {train_metric_collection} 必须为MetricCollection类型的喵! 现在的type: {type(train_metric_collection)}")
        if valid_metric_collection is None:
            valid_metric_collection = MetricCollection([])

        self.train_metric_collection = train_metric_collection
        self.valid_metric_collection = valid_metric_collection
        self.train_metrics_writer = train_metrics_writer
        self.valid_metrics_writer = valid_metrics_writer

        self.update_attempts = 0
        self.skipped_updates = 0
        self.consecutive_skipped_updates = 0
        self.data_epoch = 0

        self.task.to(self.device)

        self.global_step = 0                             # 已经调用optimizer数
        self.micro_step = 0                              # 已处理的micro batch数
        self.completed_validations = 0                   # 完成的验证数
        self.best_validation_loss = None                 # 最低的验证损失
        self.validation_rounds_without_improvement = 0   # 最低损失没有更新经过的轮数

    def _transfer_batch_device(self, batch):
        """
        迁移batch的设备到指定设备上面
        :param batch: dict
        :return: batch
        """
        if not isinstance(batch, dict):
            raise TypeError(f"batch: {batch} 必须为dict类型的喵! 现在的type: {type(batch)}")
        def move_value(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            elif isinstance(value, dict):
                result = {}
                for key, v in value.items():
                    result[key] = move_value(v)
                return result
            elif isinstance(value, list):
                result = []
                for v in value:
                    result.append(move_value(v))
                return result
            elif type(value) is tuple:
                result = []
                for v in value:
                    result.append(move_value(v))
                return tuple(result)
            else:
                return value

        moved_batch = move_value(batch)
        return moved_batch

    def _validate_task_outputs(self,outputs):
        if not isinstance(outputs, dict):
            raise TypeError(f"outputs: {outputs} 必须为dict类型喵! 现在的type: {type(outputs)}")
        if not "loss" in outputs:
            raise ValueError(f"outputs: {outputs} 中缺失了 loss 字段喵!")
        if not "losses" in outputs:
            raise ValueError(f"outputs: {outputs} 中缺失了 losses 字段喵!")
        loss = outputs["loss"]
        losses = outputs["losses"]

        def scalar_value(value, field):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{field} 必须为 torch.Tensor 喵! "
                    f"现在的类型: {type(value)}"
                )

            if not value.is_floating_point():
                raise TypeError(
                    f"{field} 必须为浮点 Tensor 喵! "
                    f"现在的 dtype: {value.dtype}"
                )

            if value.dim() != 0:
                raise ValueError(
                    f"{field} 必须为零维 Tensor 喵! "
                    f"现在的形状: {tuple(value.shape)}"
                )

            number = float(value.detach().item())

            if not isfinite(number):
                raise ValueError(
                    f"{field} 必须为有限数值，不能是 NaN 或正负无穷喵! "
                    f"收到: {number!r}"
                )

            return number

        loss_value = scalar_value(loss, 'outputs.loss')
        if not isinstance(losses, dict):
            raise TypeError(f"losses: {losses} 必须为dict类型的喵! 现在的type: {type(losses)}")
        if not losses:
            raise ValueError(f"losses 不能为空喵!")
        loss_records = {}
        for loss_name, loss_detail in losses.items():
            if not isinstance(loss_name, str):
                raise TypeError(f"loss_name: {loss_name} 必须为str类型的喵!")
            if not loss_name.strip():
                raise ValueError(f"loss_name 不能为空喵!")
            if not isinstance(loss_detail, dict):
                raise TypeError(f"loss_detail: {loss_detail} 必须为dict类型的喵! 现在的type: {type(loss_detail)}")
            if not "raw" in loss_detail:
                raise ValueError(f"loss_detail 中缺少了 raw 字段喵!")
            if not "weight" in loss_detail:
                raise ValueError(f"loss_detail 中缺少了 weight 字段喵!")
            if not "weighted" in loss_detail:
                raise ValueError(f"loss_detail 中缺少了 weighted 字段喵!")
            raw_value = scalar_value(
                loss_detail["raw"],
                f"outputs.losses.{loss_name}.raw"
            )

            weighted_value = scalar_value(
                loss_detail["weighted"],
                f"outputs.losses.{loss_name}.weighted"
            )

            weight = loss_detail["weight"]

            if type(weight) not in (int, float):
                raise TypeError(
                    f"outputs.losses.{loss_name}.weight "
                    f"必须为 Python int 或 float，不能是 bool 或 Tensor 喵! "
                    f"收到: {weight!r}，类型: {type(weight)}"
                )

            if not isfinite(weight):
                raise ValueError(
                    f"outputs.losses.{loss_name}.weight "
                    f"必须为有限数值，不能是 NaN 或正负无穷喵! "
                    f"收到: {weight!r}"
                )

            loss_records[loss_name] = {
                'raw': raw_value,
                'weight': weight,
                'weighted': weighted_value
            }

        step_record = {
            'loss': loss_value,
            'losses': loss_records
        }

        return step_record

    def _forward_backward(self, batch, *, loss_divisor, synchronize):
        if self.runtime_context.enabled and not synchronize:
            sync_context = self.train_task.no_sync()
        else:
            sync_context = nullcontext()
        with sync_context:
            outputs = None
            forward_error = None
            try:
                with self.precision.autocast_context():
                    outputs = self.train_task(batch)
            except Exception as e:
                forward_error = e
            def check_outputs():
                if forward_error is not None:
                    raise forward_error
                else:
                    step_record = self._validate_task_outputs(outputs)
                if outputs["loss"].requires_grad:
                    return step_record
                else:
                    raise ValueError(f"训练得出的 loss 不可微, 无法进行训练喵!")

            step_record = self.runtime_context.run_local('检查训练输出', check_outputs)
            backward_loss = outputs['loss'] / loss_divisor
            self.precision.backward(backward_loss)
        return outputs, step_record

    def _log_info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def _get_current_lr(self):
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]

    def _train_dry_run(self, batch):
        if not isinstance(batch, dict):
            raise ValueError(f"传入的batch必须为dict类型的喵! 现在的type: {type(batch)}")
        self.optimizer.zero_grad(set_to_none=True)
        try:
            batch = self._transfer_batch_device(batch)
            self.task.train()
            outputs = self.task(batch)
            if not isinstance(outputs, dict):
                raise ValueError(f"执行训练任务的结果必须为dict类型的喵! 现在type: {type(outputs)}")
            if not "loss" in outputs:
                raise ValueError(f"执行训练任务返回的dict必须包含 'loss' 这个键喵!")
            original_loss = outputs["loss"]
            if not isinstance(original_loss, torch.Tensor):
                raise ValueError(f"loss必须为Tensor喵! 现在的type为: {type(original_loss)}")
            if not original_loss.dim() == 0:
                raise ValueError(f"loss必须为标量型Tensor喵(即0维)! 现在的维度为: {original_loss.dim()}")
            if not torch.isfinite(original_loss).item():
                raise ValueError(f"loss出现NaN值或无限值! 训练崩了喵!")
            original_loss.backward()
            self._log_info(f"dry run已完成，未发现异常，即将开始正式训练喵!")
        except Exception as error:
            oom_info_dict = get_oom_info(error,self.device_summary,"Dry Run")
            if oom_info_dict is None:
                raise
            else:
                self._log_info(f"OutOfMemory: {oom_info_dict}")
                raise RuntimeError(f"OutOfMemory: {oom_info_dict}") from error
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def _log_memory_status(self, stage):
        context = self.runtime_context

        if self.device.type == 'cpu':
            return None

        if self.device.type == 'mps':
            if context.enabled:
                raise NotImplementedError(
                    "当前 MPS 显存监控仅保留单进程逻辑，不支持 DDP 喵!"
                )

            if not isinstance(stage, str):
                raise TypeError(
                    f"stage 必须为 str 类型的喵! 现在的type: {type(stage)}"
                )
            if stage not in ('dry_run', 'train', 'validation', 'finish'):
                raise ValueError(
                    f"stage: {stage!r} 不被允许喵! "
                    f"允许值为 dry_run、train、validation、finish"
                )

            memory_summary = get_memory_summary(self.device_summary)
            warning_summary = get_memory_warning(memory_summary)

            prefix = (
                f"stage={stage} "
                f"rank={context.rank} "
                f"device=mps "
                f"precision={self.precision.precision} "
                f"global_step={self.global_step} "
                f"micro_step={self.micro_step}"
            )
            self._log_info(
                f"memory {prefix} summary={memory_summary}"
            )

            if warning_summary is not None:
                warning_check_available = warning_summary[
                    'warning_check_available'
                ]
                recommended_max_memory = warning_summary[
                    'recommended_max_memory'
                ]

                if not warning_check_available:
                    self._log_info(
                        f"memory {prefix} "
                        f"recommended_max_memory="
                        f"{recommended_max_memory} bytes，"
                        f"无法计算 MPS 显存占用比例喵!"
                    )
                elif warning_summary['is_warning']:
                    usage_ratio = warning_summary['usage_ratio']
                    warning_threshold = warning_summary['warning_threshold']
                    driver_allocated_memory = warning_summary[
                        'driver_allocated_memory'
                    ]

                    self._log_info(
                        f"MemoryWarning {prefix} "
                        f"当前 MPS 驱动分配内存占推荐上限的比例达到阈值："
                        f"usage_ratio={usage_ratio:.1%} "
                        f"warning_threshold={warning_threshold:.1%} "
                        f"driver_allocated_memory="
                        f"{driver_allocated_memory} bytes "
                        f"recommended_max_memory="
                        f"{recommended_max_memory} bytes"
                    )

            return None

        if self.device.type != 'cuda':
            raise ValueError(
                f"不支持当前设备的显存监控喵! "
                f"device.type: {self.device.type!r}"
            )

        def collect_local():
            return self._collect_memory_record(stage)

        record = context.run_local('采集本地显存', collect_local)

        if not context.enabled:
            records = [record]
        else:
            records = [None] * context.world_size
            dist.all_gather_object(records, record)

        def write_main():
            self._write_memory_records(records)
            return None

        context.run_main('记录显存', write_main)

        return None

    def _report_local_cuda_oom(self, error, stage):
        if self.device.type != 'cuda':
            return None

        oom_error = None

        try:
            current = error

            visited = set()

            while current is not None:
                current_id = id(current)

                if current_id in visited:
                    break

                visited.add(current_id)

                if isinstance(current, torch.cuda.OutOfMemoryError):
                    oom_error = current
                    break

                current = current.__cause__

            if oom_error is None:
                return None

            if getattr(oom_error, '_lumina_memory_reported', False):
                return None

            oom_error._lumina_memory_reported = True

            device_index = self.device.index
            if device_index is None:
                device_index = self.device_summary['device_index']

            diagnostic = {
                'stage': stage,
                'rank': self.runtime_context.rank,
                'device_index': device_index,
                'global_step': self.global_step,
                'micro_step': self.micro_step,
                'precision': self.precision.precision,
                'error_type': type(oom_error).__name__,
                'message': str(oom_error),
                'memory': None,
                'advice': None,
            }

            try:
                diagnostic['advice'] = get_oom_info(
                    oom_error,
                    self.device_summary,
                    stage,
                )
            except Exception as advice_error:
                diagnostic['advice_error'] = (
                    f"{type(advice_error).__name__}: {advice_error}"
                )

            try:
                diagnostic['memory'] = get_memory_summary(
                    self.device_summary
                )
            except Exception as memory_error:
                diagnostic['memory_error'] = (
                    f"{type(memory_error).__name__}: {memory_error}"
                )

            message = f"CUDA OOM: {diagnostic}"
            context = self.runtime_context

            if context.is_main and self.logger is not None:
                self._log_info(message)
            else:
                print(message, flush=True)

        except Exception as diagnostic_error:
            if oom_error is not None:
                note_target = oom_error
            else:
                note_target = error

            try:
                note_target.add_note(
                    f"获取或输出 CUDA OOM 诊断时发生异常："
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
            except Exception:
                pass

            try:
                print(
                    f"CUDA OOM 诊断失败："
                    f"rank={self.runtime_context.rank} "
                    f"stage={stage} "
                    f"error_type={type(diagnostic_error).__name__} "
                    f"message={diagnostic_error}",
                    flush=True,
                )
            except Exception:
                pass

        return None

    def train_step(self, batch, should_zero_grad, should_update, loss_divisor):
        """
        单步训练方法
        :param batch: dict {
        例如MlmTask:
        "input_ids": LongTensor, [batch_size, seq_length],
        "attention_mask": LongTensor, [batch_size, seq_length],
        "token_type_ids": LongTensor, [batch_size, seq_length],
        "labels": LongTensor, [batch_size, seq_length],
        }
        :param should_zero_grad: bool, 是否清除梯度
        :param should_update: bool, 是否更新参数
        :param loss_divisor: int, loss缩放参数
        :return: new_dict, 执行训练任务后脱离计算图的输出结果
        """
        def prepare_batch():
            if not type(should_zero_grad) is bool:
                raise TypeError(f"should_zero_grad: {should_zero_grad} 必须为bool值喵! 现在的type:{type(should_zero_grad)}")
            if not type(should_update) is bool:
                raise TypeError(f"should_update: {should_update} 必须为bool值喵! 现在的type:{type(should_update)}")
            if not type(loss_divisor) is int or loss_divisor <= 0:
                raise TypeError(f"loss_divisor: {loss_divisor} 必须为正整数的喵!")
            self.train_task.train()
            moved_batch = self._transfer_batch_device(batch)
            if should_zero_grad:
                self.optimizer.zero_grad(set_to_none=True)
            return moved_batch
        operation = 'train.prepare'
        try:
            moved_batch = self.runtime_context.run_local('准备训练批次', prepare_batch)
            operation = 'train.forward_backward'
            outputs, step_record = self._forward_backward(moved_batch, loss_divisor=loss_divisor, synchronize=should_update)
            def update_metrics():
                with torch.no_grad():
                    self.train_metric_collection.update(batch=moved_batch, outputs=outputs)
                return None
            operation = 'train.metrics'
            self.runtime_context.run_local('更新训练指标', update_metrics)
            did_update = False
            if should_update:
                operation = 'train.optimizer'
                did_update = self.precision.step(self.optimizer, self.runtime_context)
                def clear_cycle_gradients():
                    self.optimizer.zero_grad(set_to_none=True)
                    return None
                operation = 'train.clear_gradients'
                self.runtime_context.run_local('清理周期梯度', clear_cycle_gradients)
            step_record['did_update'] = did_update
            return step_record
        except Exception as error:
            self._report_local_cuda_oom(error, operation)
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as cleanup_error:
                error.add_note(
                    f"处理训练异常时，清理梯度也失败了："
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

            raise

    def _record_train_step(self, step_result, *, should_update, run_accumulator, interval_accumulator):
        MAX_CONSECUTIVE_SKIPPED_UPDATES = 100
        context = self.runtime_context
        def record_batch():
            did_update = step_result["did_update"]

            if type(did_update) is not bool:
                raise TypeError(
                    f"step_result['did_update'] 必须为 bool 类型喵! "
                    f"收到: {did_update!r}，类型: {type(did_update)}"
                )

            if not should_update and did_update:
                raise ValueError(
                    "梯度累积周期的中间批不能更新参数喵! "
                    f"should_update={should_update}，did_update={did_update}"
                )

            run_accumulator.update(step_result)
            interval_accumulator.update(step_result)

            self.micro_step += 1

            return step_result["loss"]

        last_loss = context.run_local('累计训练批次', record_batch)
        if not should_update:
            return last_loss
        did_update = step_result['did_update']
        def record_update():
            self.update_attempts += 1
            if did_update:
                self.global_step += 1
                self.consecutive_skipped_updates = 0
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                self.skipped_updates += 1
                self.consecutive_skipped_updates += 1
            if self.consecutive_skipped_updates >= MAX_CONSECUTIVE_SKIPPED_UPDATES:
                scale = None
                if self.precision.scaler is not None:
                    scale = self.precision.scaler.get_scale()

                raise RuntimeError(
                    f"连续跳过参数更新的次数已达到上限，停止训练喵!\n"
                    f"precision: {self.precision.precision}\n"
                    f"global_step: {self.global_step}\n"
                    f"连续跳过次数: {self.consecutive_skipped_updates}\n"
                    f"允许上限: {MAX_CONSECUTIVE_SKIPPED_UPDATES}\n"
                    f"当前缩放因子 scale: {scale}\n"
                    f"请检查损失与梯度是否持续出现 NaN 或无穷值。"
                )

            return None
        context.run_local('记录参数更新', record_update)
        return last_loss

    def _collect_train_summary(self,run_accumulator,interval_accumulator,*,include_metrics=True):
        run_summary = run_accumulator.summarize(self.runtime_context)
        interval_summary = interval_accumulator.summarize(self.runtime_context)
        metrics = {}
        if include_metrics and interval_summary['completed_batches'] > 0:
            metrics = self.train_metric_collection.compute_distributed(self.runtime_context)
        train_summary = {'run': run_summary, 'interval': interval_summary, 'metrics': metrics}
        return train_summary

    def _collect_valid_summary(self, accumulator, *, include_metrics=True):
        run_summary = accumulator.summarize()
        metrics = {}
        if include_metrics:
            metrics = self.valid_metric_collection.compute()
        valid_summary = {'run': run_summary, 'metrics': metrics}
        return valid_summary

    def _make_metrics_row(self,summary,metric_values,interval_summary=None):
        row = {'global_step': self.global_step, 'avg_loss': summary['avg_loss']}
        if interval_summary is not None:
            row['lr'] = self._get_current_lr()
            row['interval_avg_loss'] = interval_summary['avg_loss']
        for loss_name, loss_detail in summary["losses"].items():
            row[f'{loss_name}_avg_raw'] = loss_detail['avg_raw']
            row[f'{loss_name}_avg_weighted_loss'] = loss_detail['avg_weighted']
        if interval_summary is not None:
            for loss_name, loss_detail in interval_summary["losses"].items():
                row[f'{loss_name}_interval_avg_raw'] = loss_detail['avg_raw']
                row[f'{loss_name}_interval_avg_weighted_loss'] = loss_detail['avg_weighted']
        for key, value in metric_values.items():
            if key in row:
                raise ValueError(f"row: {row} 中存在重复列名: {key} 喵!")
            row[key] = value
        return row

    def _write_train_summary(self,train_summary):
        if not train_summary['interval']['completed_batches']>0:
            raise ValueError(f"completed_batches: {train_summary['interval']['completed_batches']} 不大于0时无法进行写入操作喵!")
        row = self._make_metrics_row(train_summary['run'], train_summary['metrics'], interval_summary=train_summary['interval'])
        if self.train_metrics_writer is not None:
            self.train_metrics_writer.write(row)
        return None

    def _finish_train_interval(self,interval_accumulator):
        def reset_interval():
            interval_accumulator.reset()
            self.train_metric_collection.reset()
            return None
        self.runtime_context.run_local('清空已报告训练区间', reset_interval)
        return None

    def valid_step(self, batch):
        """
        单步验证方法
        :param batch: dict {
        例如MlmTask:
        "input_ids": LongTensor, [batch_size, seq_length],
        "attention_mask": LongTensor, [batch_size, seq_length],
        "token_type_ids": LongTensor, [batch_size, seq_length],
        "labels": LongTensor, [batch_size, seq_length],
        }
        :return: new_dict, 执行训练任务后脱离计算图的输出结果
        """
        moved_batch=self._transfer_batch_device(batch)
        self.task.eval()
        with torch.no_grad():
            with self.precision.autocast_context():
                outputs = self.task(moved_batch)
            step_record = self._validate_task_outputs(outputs)
            self.valid_metric_collection.update(batch=moved_batch, outputs=outputs)
        return step_record


    def validate(self, valid_dataloader):
        """
        遍历验证dataloader的batch的方法
        :param valid_dataloader: 验证数据加载器
        :return: summary_dict = {
                "completed_batches": completed_batches,
                "last_loss": last_loss,
                "avg_loss": avg_loss,
            }
        """
        if valid_dataloader is None:
            raise ValueError(f"valid_dataloader 不能为空喵!")
        accumulator = LossAccumulator()
        self.valid_metric_collection.reset()
        log_every_batches = 100
        total_batches = len(valid_dataloader)
        for batch in valid_dataloader:
            step_record = self.valid_step(batch)
            accumulator.update(step_record)
            completed_batches = accumulator.batch_count
            needs_log = (
                completed_batches == 1
                or completed_batches % log_every_batches == 0
                or completed_batches == total_batches
            )

            if needs_log:
                log_summary = self._collect_valid_summary(
                    accumulator,
                    include_metrics=True,
                )

                def log_only():
                    message = (
                        f"[valid] step={self.global_step}, "
                        f"batch={log_summary['run']['completed_batches']} / {total_batches}, "
                        f"loss={log_summary['run']['avg_loss']}, "
                        f"metrics={log_summary['metrics']}"
                    )
                    self._log_info(message)
                    return None

                log_only()
        if accumulator.batch_count == 0:
            raise ValueError("DataLoader 未产生批次, completed_batches 为0了喵!")

        valid_summary = self._collect_valid_summary(
            accumulator,
            include_metrics=True,
        )
        loss_summary = valid_summary["run"]
        valid_metrics = valid_summary["metrics"]
        validation_summary = {
            'avg_loss': loss_summary['avg_loss'],
            'losses': loss_summary['losses'],
            'completed_batches': loss_summary['completed_batches'],
            'valid_metrics': valid_metrics
        }
        return validation_summary

    def _validation_event(self, valid_dataloader, train_summary, interval_accumulator, validation_options, checkpoint_options):
        context = self.runtime_context
        def main_action():
            if not valid_dataloader:
                raise ValueError(f"传入的valid_dataloader不能为空喵!")
            validation_summary = self.validate(valid_dataloader)
            current_loss = validation_summary['avg_loss']
            if not validation_summary['completed_batches']>0:
                raise ValueError(f"completed_batches: {validation_summary['completed_batches']} 必须大于0喵!")
            improved = self.best_validation_loss is None or current_loss < self.best_validation_loss
            self.completed_validations += 1
            if improved:
                self.best_validation_loss = current_loss
                self.validation_rounds_without_improvement = 0
            else:
                self.validation_rounds_without_improvement += 1
            patience = validation_options['patience']
            early_stopped = False
            if patience:
                early_stopped = self.validation_rounds_without_improvement >= patience
            saved_paths = []
            if checkpoint_options['enabled']:
                directory = checkpoint_options['directory']
                save_every_rounds = checkpoint_options['save_every_rounds']
                save_best = improved
                save_last = self.completed_validations % save_every_rounds == 0
                if save_best or save_last:
                    checkpoint = build_checkpoint(self.task, self.optimizer, self.scheduler, self)
                    if save_best:
                        best_path = directory / 'best.pt'
                        save_checkpoint(checkpoint, best_path)
                        saved_paths.append(str(best_path))
                    if save_last:
                        last_path = directory / 'last.pt'
                        save_checkpoint(checkpoint, last_path)
                        saved_paths.append(str(last_path))
            self._write_train_summary(train_summary)
            valid_row = self._make_metrics_row(validation_summary, validation_summary['valid_metrics'], interval_summary=None)
            if self.valid_metrics_writer is not None:
                self.valid_metrics_writer.write(valid_row)
            progress = self.get_training_progress()
            event_result = {'validation_summary': validation_summary, 'progress': progress, 'early_stopped': early_stopped, 'saved_paths': saved_paths}
            return event_result
        event_result = context.run_main('验证与保存', main_action)
        def apply_progress():
            self.load_training_progress(event_result['progress'])
        context.run_local('同步验证进度', apply_progress)
        self._finish_train_interval(interval_accumulator)
        return event_result

    def _finish_training(self, run_accumulator, interval_accumulator, *, last_loss, last_train_summary, last_validation_summary, early_stopped, checkpoint_options, no_training_needed=False):
        context = self.runtime_context
        avg_loss = None
        last_interval_avg_loss = None
        global_last_loss = None
        losses = {}
        last_interval_losses = {}
        train_metrics = {}
        if no_training_needed:
            last_validation_summary = None
            early_stopped = False
        else:
            def check_final_inputs():
                if not isinstance(run_accumulator, LossAccumulator):
                    raise TypeError(f"run_accumulator: {run_accumulator} 必须为LossAccumulator类型的喵! 现在的type: {type(run_accumulator)}")
                if not isinstance(interval_accumulator, LossAccumulator):
                    raise TypeError(f"interval_accumulator: {interval_accumulator} 必须为LossAccumulator类型的喵! 现在的type: {type(interval_accumulator)}")
                if not run_accumulator.batch_count>0:
                    raise ValueError(f"batch_count: {run_accumulator.batch_count} 必须至少为1喵!")
                if not type(last_loss) in (int, float) or not isfinite(last_loss):
                    raise ValueError(f"last_loss: {last_loss} 必须为有限数值喵!")
                return interval_accumulator.batch_count>0
            has_interval = context.run_local('检查训练收尾', check_final_inputs)
            if context.enabled:
                interval_flags = [None] * context.world_size
                dist.all_gather_object(interval_flags, has_interval)
                reference = interval_flags[0]

                for rank, flag in enumerate(interval_flags):
                    if flag != reference:
                        raise RuntimeError(
                            "训练收尾时，各进程的 has_interval 不一致喵! "
                            f"rank 0 为 {reference}，rank {rank} 为 {flag}；"
                            f"全部结果为 {interval_flags}。"
                            "请检查各进程的区间累计和清空时机是否一致。"
                        )
            if has_interval:
                last_train_summary = self._collect_train_summary(run_accumulator, interval_accumulator,  include_metrics=True)
                def write_tail():
                    self._write_train_summary(last_train_summary)
                context.run_main('写末尾训练统计', write_tail)
                self._finish_train_interval(interval_accumulator)
            run_summary = run_accumulator.summarize(context)
            avg_loss = run_summary['avg_loss']
            losses = run_summary['losses']
            if last_train_summary is not None:
                interval_summary = last_train_summary['interval']
                last_interval_avg_loss = interval_summary['avg_loss']
                last_interval_losses = interval_summary['losses']
                train_metrics = last_train_summary['metrics']
                global_last_loss = last_loss
                if context.enabled:
                    reduce_device = context.device
                    last_loss_tensor = torch.tensor(
                        last_loss,
                        dtype=torch.float64,
                        device=reduce_device,
                    )
                    dist.all_reduce(
                        last_loss_tensor,
                        op=dist.ReduceOp.SUM,
                    )
                    global_last_loss = (
                            float(last_loss_tensor.item())
                            / context.world_size
                    )

                if checkpoint_options['enabled']:
                    def save_final():
                        checkpoint = build_checkpoint(
                            self.task,
                            self.optimizer,
                            self.scheduler,
                            self,
                        )
                        last_path = (
                                checkpoint_options['directory'] / 'last.pt'
                        )
                        save_checkpoint(checkpoint, last_path)
                        return str(last_path)

                    context.run_main('保存最终权重', save_final)
        summary = {
            'completed_iters': self.global_step,
            'global_step': self.global_step,
            'micro_step': self.micro_step,
            'avg_loss': avg_loss,
            'last_loss': global_last_loss,
            'losses': losses,
            'last_interval_losses': last_interval_losses,
            'completed_validations': self.completed_validations,
            'last_validation_summary': last_validation_summary,
            'best_validation_loss': self.best_validation_loss,
            'validation_rounds_without_improvement': self.validation_rounds_without_improvement,
            'early_stopped': early_stopped,
            'train_metrics': train_metrics,
            'last_interval_avg_loss': last_interval_avg_loss,
            'no_training_needed': no_training_needed,
            'update_attempts': self.update_attempts,
            'skipped_updates': self.skipped_updates,
        }
        return summary

    def fit(self, train_dataloader, total_steps, log_every_steps=100, accumulation_steps=1, valid_dataloader=None, validation_enabled=False, validation_every_steps=None, early_stopping_patience=None, checkpoint_save_every_rounds=None, checkpoints_dir = None ):
        """
        fit方法负责不断从dataloader中取出数据，并调用train_step方法
        :param train_dataloader: 训练数据加载器
        :param total_steps: 需要完成的总迭代次数
        :param log_every_steps: 每隔多少个iter打印一次日志
        :param accumulation_steps: 梯度积累
        :param valid_dataloader: 验证数据加载器
        :param validation_enabled: 是否进行验证
        :param validation_every_steps: 每隔多少步验证一次
        :param early_stopping_patience: 早停的patience
        :param checkpoint_save_every_rounds: 每验证多少轮保存一次检查点
        :param checkpoints_dir: ckpt检查点保存路径
        :return: summary_dict = {
                            "completed_iters": self.global_step,
                            "global_step": self.global_step,
                            "micro_step": self.micro_step,
                            "avg_loss": avg_loss,
                            "last_loss": last_loss,
                            "completed_validations": self.completed_validations,
                            "last_validation_summary": last_validation_summary,
                            "best_validation_loss": self.best_validation_loss,
                            "validation_rounds_without_improvement": self.validation_rounds_without_improvement,
                            "early_stopped": early_stopped,
                            "checkpoint_save_every_rounds": checkpoint_save_every_rounds,
                        }
        """
        context = self.runtime_context
        def check_fit_options():
            if not type(total_steps) is int or total_steps <= 0:
                raise ValueError(f"total_steps必须为大于0的整数喵! 现在是: {total_steps}")
            if not type(log_every_steps) is int or log_every_steps <= 0:
                raise ValueError(f"log_every_steps必须为大于0的整数喵! 现在是: {log_every_steps}")
            if not type(accumulation_steps) is int or accumulation_steps <= 0:
                raise ValueError(f"accumulation_steps必须为大于0的整数喵! 现在是: {accumulation_steps}")
            if not type(validation_enabled) is bool:
                raise ValueError(f"validation_enabled必须为bool类型的喵! 现在的type: {type(validation_enabled)}")
            if validation_enabled:
                if valid_dataloader is None and context.is_main:
                    raise ValueError(f"valid_dataloader在validation_enabled时不能为空喵!")
                if not type(validation_every_steps) is int or validation_every_steps <= 0:
                    raise ValueError(f"validation_every_steps必须为正整数喵! 现在是: {validation_every_steps}")
                if early_stopping_patience is not None and (type(early_stopping_patience) is not int or early_stopping_patience <= 0):
                    raise ValueError(f"early_stopping_patience必须为正整数喵! 现在是: {early_stopping_patience}")
                if (checkpoints_dir is None) != (checkpoint_save_every_rounds is None):
                    raise ValueError(
                        "checkpoints_dir 与 checkpoint_save_every_rounds "
                        "必须同时为 None 或同时非 None 喵!\n"
                        f"checkpoints_dir: {checkpoints_dir!r}\n"
                        f"checkpoint_save_every_rounds: {checkpoint_save_every_rounds!r}"
                    )

                directory = None

                if checkpoints_dir is not None:
                    if not isinstance(checkpoints_dir, (str, pathlib.Path)):
                        raise TypeError(
                            "checkpoints_dir 必须为 str 或 pathlib.Path 喵! "
                            f"当前类型为 {type(checkpoints_dir)}"
                        )

                    if isinstance(checkpoints_dir, str) and not checkpoints_dir.strip():
                        raise ValueError("checkpoints_dir 不能是空字符串或全空白字符串喵!")

                    directory = pathlib.Path(checkpoints_dir)
                    if not directory.is_dir():
                        raise ValueError(
                            f"checkpoints_dir 必须指向已经存在的目录喵! 当前路径: {directory}"
                        )

                    if type(checkpoint_save_every_rounds) is not int or checkpoint_save_every_rounds <= 0:
                        raise ValueError(
                            "checkpoint_save_every_rounds 必须为正整数喵! "
                            f"当前值: {checkpoint_save_every_rounds!r}"
                        )

                validation_options = {
                    'patience': early_stopping_patience if validation_enabled else None,
                }

                checkpoint_options = {
                    'enabled': checkpoints_dir is not None,
                    'save_every_rounds': checkpoint_save_every_rounds,
                    'directory': directory,
                }

                settings = {
                    'total_steps': total_steps,
                    'log_every_steps': log_every_steps,
                    'accumulation_steps': accumulation_steps,
                    'validation_enabled': validation_enabled,
                    'validation_every_steps': validation_every_steps,
                    'early_stopping_patience': early_stopping_patience,
                    'checkpoint_save_every_rounds': checkpoint_save_every_rounds,
                    'checkpoint_enabled': checkpoint_options['enabled'],
                    'checkpoint_directory': str(directory) if directory is not None else None,
                    'progress': self.get_training_progress(),
                    'data_epoch': self.data_epoch,
                    'precision': self.precision.precision,
                }
                return (validation_options,checkpoint_options,settings)

        validation_options, checkpoint_options, settings = context.run_local('检查训练循环参数', check_fit_options)
        if context.enabled:
            settings_list = [None] * context.world_size
            dist.all_gather_object(settings_list, settings)
            reference = settings_list[0]

            for rank, rank_settings in enumerate(settings_list):
                missing_keys = set(reference) - set(rank_settings)
                extra_keys = set(rank_settings) - set(reference)

                if missing_keys or extra_keys:
                    raise ValueError(
                        f"训练设置字段不一致喵! rank {rank} 相对于 rank 0：\n"
                        f"缺失字段: {sorted(missing_keys)}\n"
                        f"多余字段: {sorted(extra_keys)}"
                    )

                for key in sorted(reference):
                    reference_value = reference[key]
                    rank_value = rank_settings[key]

                    if rank_value != reference_value:
                        raise ValueError(
                            f"训练设置不一致喵! 字段: {key!r}；"
                            f"rank 0 的值: {reference_value!r}；"
                            f"rank {rank} 的值: {rank_value!r}"
                        )

        if self.global_step>=total_steps:
            summary = self._finish_training(None, None, last_loss=None, last_train_summary=None, last_validation_summary=None, early_stopped=False, checkpoint_options=checkpoint_options, no_training_needed=True)
            return summary

        try:
            initial_global_step = self.global_step
            if self.device.type == 'cuda':
                def reset_local():
                    reset_peak_memory_stats(self.device_summary)
                context.run_local("重置本次训练显存峰值", reset_local)
            run_accumulator = LossAccumulator()
            interval_accumulator = LossAccumulator()
            last_loss = None
            last_train_summary = None
            last_validation_summary = None
            early_stopped = False
            def prepare_training():
                self.train_metric_collection.reset()
                batch_num = len(train_dataloader)
                if not type(batch_num) is int or batch_num <= 0:
                    raise ValueError(f"batch_num: {batch_num} 必须为正整数喵!")
                return batch_num
            batch_num = context.run_local('准备训练统计与长度', prepare_training)
            if context.enabled:
                batch_counts = [None] * context.world_size
                dist.all_gather_object(batch_counts, batch_num)
                reference = batch_counts[0]
                for rank, batch_count in enumerate(batch_counts):
                    if batch_count != reference:
                        raise ValueError(f"各进程的batch_count不一致喵!\n"
                                         f"rank 0 为: {reference}, rank: {rank} 为 {batch_count}\n"
                                         f"请检查各进程batch_count是否一致")
            def prepare_dry_run():
                if context.enabled:
                    if not isinstance(train_dataloader.sampler,DistributedSampler):
                        raise TypeError(f"train_dataloader.sampler: {train_dataloader.sampler} 必须为 DistributedSampler 类型的喵! 现在的type: {type(train_dataloader.sampler)}")
                    train_dataloader.sampler.set_epoch(self.data_epoch)
                dry_iterator = iter(train_dataloader)
                dry_batch = next(dry_iterator)
                moved_batch = self._transfer_batch_device(dry_batch)
                self.train_task.train()
                self.optimizer.zero_grad(set_to_none=True)
                return moved_batch

            primary_error = None

            try:
                dry_batch = context.run_local(
                    '准备试运行批次',
                    prepare_dry_run,
                )
                self._forward_backward(
                    dry_batch,
                    loss_divisor=1,
                    synchronize=True,
                )

            except Exception as error:
                primary_error = error
                self._report_local_cuda_oom(error, 'dry_run')

                raise

            finally:
                try:
                    self.optimizer.zero_grad(set_to_none=True)

                except Exception as cleanup_error:
                    if primary_error is None:
                        raise

                    try:
                        primary_error.add_note(
                            f"试运行失败后，清理梯度也发生异常："
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    except Exception:
                        pass

            self._log_memory_status('dry_run')
            def make_iterator():
                sampler = train_dataloader.sampler

                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(self.data_epoch)

                created_iterator = iter(train_dataloader)
                return created_iterator

            def read_batch():
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        f"训练数据提前结束喵! "
                        f"本轮预期 {batch_num} 个 batch，"
                        f"已处理 {idx} 个，"
                        f"读取第 {idx + 1} 个时迭代器已经耗尽。"
                    ) from error

                return batch

            while self.global_step < total_steps and not early_stopped:
                iterator = context.run_local('建立本轮数据迭代器', make_iterator)
                idx = 0
                while idx < batch_num and self.global_step < total_steps and not early_stopped:
                    current_cycle_size = min(accumulation_steps, batch_num - idx)
                    accumulated_batches = 0
                    while accumulated_batches < current_cycle_size:
                        batch = context.run_local('读取训练批次',read_batch)
                        should_zero_grad = accumulated_batches == 0
                        should_update = accumulated_batches + 1 == current_cycle_size
                        step_result = self.train_step(batch, should_zero_grad=should_zero_grad, should_update=should_update, loss_divisor=current_cycle_size)
                        last_loss = self._record_train_step(step_result, should_update=should_update, run_accumulator=run_accumulator, interval_accumulator=interval_accumulator)
                        idx += 1
                        accumulated_batches += 1
                    if not step_result['did_update']:
                        pass
                    else:
                        needs_validation = self.global_step % validation_every_steps==0 or self.global_step>=total_steps if validation_enabled else False
                        needs_log = self.global_step % log_every_steps == 0
                        first_update = (
                                self.global_step == initial_global_step + 1
                        )

                        if first_update or needs_log:
                            self._log_memory_status('train')

                        if needs_validation:
                            train_summary = self._collect_train_summary(run_accumulator, interval_accumulator, include_metrics=True)
                            event_result = self._validation_event(valid_dataloader, train_summary, interval_accumulator, validation_options, checkpoint_options)
                            last_train_summary = train_summary
                            last_validation_summary = event_result['validation_summary']
                            early_stopped = event_result['early_stopped']
                            self._log_memory_status('validation')
                            if needs_log:
                                def log_event():
                                    message = f"step={self.global_step} / {total_steps}, loss={train_summary['run']['avg_loss']}, interval_loss={train_summary['interval']['avg_loss']}, lr={self._get_current_lr()}"
                                    self._log_info(message)
                                    return None
                                context.run_main('记录训练摘要', log_event)
                        elif needs_log:
                            log_summary = self._collect_train_summary(run_accumulator, interval_accumulator, include_metrics=False)
                            def log_only():
                                message = f"step={self.global_step}, loss={log_summary['run']['avg_loss']}, interval_loss={log_summary['interval']['avg_loss']}, lr={self._get_current_lr()}"
                                self._log_info(message)
                                return None
                            context.run_main('记录训练摘要', log_only)
                if self.global_step<total_steps and not early_stopped:
                    self.data_epoch += 1
            summary = self._finish_training(run_accumulator, interval_accumulator, last_loss=last_loss,
                                        last_train_summary=last_train_summary,
                                        last_validation_summary=last_validation_summary, early_stopped=early_stopped,
                                        checkpoint_options=checkpoint_options, no_training_needed=False)
            return summary
        except Exception as error:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
            raise

    def get_training_progress(self):
        """
        获取训练进度状态
        :return: training_dict
        """
        training_dict = {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "completed_validations": self.completed_validations,
            "best_validation_loss": self.best_validation_loss,
            "validation_rounds_without_improvement": self.validation_rounds_without_improvement,
            "update_attempts": self.update_attempts,
            "skipped_updates": self.skipped_updates,
            "consecutive_skipped_updates": self.consecutive_skipped_updates
        }
        return training_dict

    def load_training_progress(self, training_dict):
        """
        加载训练进度状态
        :param training_dict: 训练状态参数
        :return:
        """

        validated_progress = _validate_training_progress(training_dict)

        global_step = validated_progress["global_step"]
        micro_step = validated_progress["micro_step"]
        completed_validations = validated_progress["completed_validations"]
        best_validation_loss = validated_progress["best_validation_loss"]
        validation_rounds_without_improvement = validated_progress["validation_rounds_without_improvement"]
        update_attempts = validated_progress["update_attempts"]
        skipped_updates = validated_progress["skipped_updates"]
        consecutive_skipped_updates = validated_progress["consecutive_skipped_updates"]

        self.global_step = global_step
        self.micro_step = micro_step
        self.completed_validations = completed_validations
        self.best_validation_loss = best_validation_loss
        self.validation_rounds_without_improvement = validation_rounds_without_improvement
        self.update_attempts = update_attempts
        self.skipped_updates = skipped_updates
        self.consecutive_skipped_updates = consecutive_skipped_updates

    def _collect_memory_record(self, stage):
        if not isinstance(stage, str):
            raise TypeError(f"stage: {stage} 必须为str类型的喵! 现在的type: {type(stage)}")
        if not stage.strip():
            raise ValueError(f"stage 不能为空喵!")
        allowed_stages = ["dry_run", "train", "validation", "finish"]
        if stage not in allowed_stages:
            raise ValueError(f"当前stage: {stage} 不被允许喵! 允许的stage: {allowed_stages}")
        if self.device.type == 'cpu':
            return None
        elif self.device.type == 'cuda':
            summary = self.device_summary
            if not isinstance(summary, dict):
                raise TypeError(f"summary: {summary} 必须为dict类型的喵! 现在的type: {type(summary)}")
            if not 'backend' in summary:
                raise ValueError(f"summary: {summary} 中缺少了 backend 字段喵!")
            if not 'gpu_nums' in summary:
                raise ValueError(f"summary: {summary} 中缺少了 gpu_nums 字段喵!")
            if not 'device_index' in summary:
                raise ValueError(f"summary: {summary} 中缺少了 device_index 字段喵!")
            backend = summary['backend']
            gpu_nums = summary['gpu_nums']
            device_index = summary['device_index']
            if not backend == 'cuda':
                raise ValueError(f"device_type: {self.device.type} 为cuda时, backend: {backend} 必须为cuda喵!")
            if not type(gpu_nums) is int or not gpu_nums == 1:
                raise ValueError(f"gpu_nums: {gpu_nums} 必须严格等于1喵!")
            if not type(device_index) is int or device_index < 0:
                raise ValueError(f"device_index: {device_index} 必须为非负整数喵!")
            if self.device.index is not None:
                if self.device.index != device_index:
                    raise ValueError(f"device.index: {self.device.index} 必须与device_index: {device_index} 相等喵!")
            memory = get_memory_summary(summary)
            if not isinstance(memory, dict):
                raise TypeError(f"memory: {memory} 必须为dict类型的喵! 现在的type: {type(memory)}")
            memory_backend = memory['backend']
            memory_gpu_nums = memory['gpu_nums']
            memory_device_index = memory['device_index']
            if not memory_backend == 'cuda':
                raise ValueError(f"memory_backend: {memory_backend} 必须为cuda喵!")
            if not type(memory_gpu_nums) is int or not memory_gpu_nums == 1:
                raise ValueError(f"memory_gpu_nums: {memory_gpu_nums} 必须为1喵!")
            if not memory_device_index == device_index:
                raise ValueError(f"memory_device_index: {memory_device_index} 必须与 device_index: {device_index} 相等喵!")
            if not 'allocated_memory' in memory:
                raise ValueError(f"memory: {memory} 缺少了 allocated_memory 字段喵!")
            allocated_memory = memory['allocated_memory']
            if not 'reserved_memory' in memory:
                raise ValueError(f"memory: {memory} 缺少了 reserved_memory 字段喵!")
            reserved_memory = memory['reserved_memory']
            if not 'max_allocated_memory' in memory:
                raise ValueError(f"memory: {memory} 缺少了 max_allocated_memory 字段喵!")
            max_allocated_memory = memory['max_allocated_memory']
            if not 'max_reserved_memory' in memory:
                raise ValueError(f"memory: {memory} 缺少了 max_reserved_memory 字段喵!")
            max_reserved_memory = memory['max_reserved_memory']
            if not 'free_memory' in memory:
                raise ValueError(f"memory: {memory} 缺少了 free_memory 字段喵!")
            free_memory = memory['free_memory']
            if not 'total_memory' in memory:
                raise ValueError(f"memory: {memory} 缺少了 total_memory 字段喵!")
            total_memory = memory['total_memory']
            if not type(allocated_memory) is int:
                raise TypeError(f"allocated_memory: {allocated_memory} 必须为int类型的喵! 现在的type: {type(allocated_memory)}")
            if not type(reserved_memory) is int:
                raise TypeError(f"reseved_memory: {reserved_memory} 必须为int类型的喵! 现在的type: {type(reserved_memory)}")
            if not type(max_allocated_memory) is int:
                raise TypeError(f"max_allocated_memory: {max_allocated_memory} 必须为int类型的喵! 现在的type: {type(max_allocated_memory)}")
            if not type(max_reserved_memory) is int:
                raise TypeError(f"max_reserved_memory: {max_reserved_memory} 必须为int类型的喵! 现在的type: {type(max_reserved_memory)}")
            if not type(free_memory) is int:
                raise TypeError(f"free_memory: {free_memory} 必须为int类型的喵! 现在的type: {type(free_memory)}")
            if not type(total_memory) is int:
                raise TypeError(f"total_memory: {total_memory} 必须为int类型的喵! 现在的type: {type(total_memory)}")
            if not total_memory > 0:
                raise ValueError(f"total_memory: {total_memory} 必须大于0喵!")
            if not allocated_memory >= 0:
                raise ValueError(f"allocated_memory: {allocated_memory} 必须为非负数喵!")
            if not reserved_memory >= 0:
                raise ValueError(f"reserved_memory: {reserved_memory} 必须为非负数喵!")
            if not max_allocated_memory >= 0:
                raise ValueError(f"max_allocated_memory: {max_allocated_memory} 必须为非负数喵!")
            if not max_reserved_memory >= 0:
                raise ValueError(f"max_reserved_memory: {max_reserved_memory} 必须为非负数喵!")
            if not free_memory >= 0:
                raise ValueError(f"free_memory: {free_memory} 必须为非负数喵!")
            warning = get_memory_warning(memory, warning_threshold=0.9)
            if not isinstance(warning, dict):
                raise TypeError(f"warning: {warning} 必须为dict类型的喵! 现在的type: {type(warning)}")
            record = {
                'stage': stage,
                'rank': self.runtime_context.rank,
                'device_index': memory_device_index,
                'global_step': self.global_step,
                'micro_step': self.micro_step,
                'precision': self.precision.precision,
                'memory': memory,
                'warning': warning,
            }
            return record
        else:
            raise ValueError(f"当前框架只支持cpu与cuda喵!")

    def _write_memory_records(self, records):
        if not isinstance(records, list):
            raise TypeError(f"records: {records} 必须为 list 类型的喵! 现在的type: {type(records)}")
        if not records:
            raise ValueError(f"records 不能为空喵!")
        lines = []
        for record in records:
            stage = record['stage']
            rank = record['rank']
            device_index = record['device_index']
            global_step = record['global_step']
            micro_step = record['micro_step']
            precision = record['precision']
            memory = record['memory']
            warning = record['warning']
            allocated_gib = memory['allocated_memory'] / (1024 ** 3)
            reserved_gib = memory['reserved_memory'] / (1024 ** 3)
            peak_allocated_gib = memory['max_allocated_memory'] / (1024 ** 3)
            peak_reserved_gib = memory['max_reserved_memory'] / (1024 ** 3)
            free_gib = memory['free_memory'] / (1024 ** 3)
            total_gib = memory['total_memory'] / (1024 ** 3)

            message = (
                f"memory "
                f"stage={stage} "
                f"rank={rank} "
                f"device=cuda:{device_index} "
                f"precision={precision} "
                f"global_step={global_step} "
                f"micro_step={micro_step} "
                f"allocated={allocated_gib:.3f} GiB "
                f"reserved={reserved_gib:.3f} GiB "
                f"peak_allocated={peak_allocated_gib:.3f} GiB "
                f"peak_reserved={peak_reserved_gib:.3f} GiB "
                f"free={free_gib:.3f} GiB "
                f"total={total_gib:.3f} GiB"
            )
            lines.append(message)
            if warning['is_warning']:
                usage_ratio = warning['usage_ratio']
                warning_threshold = warning['warning_threshold']

                warning_message = (
                    f"MemoryWarning "
                    f"stage={stage} "
                    f"rank={rank} "
                    f"device=cuda:{device_index} "
                    f"precision={precision} "
                    f"global_step={global_step} "
                    f"micro_step={micro_step} "
                    f"本次 fit 保留显存累计峰值达到阈值："
                    f"usage_ratio={usage_ratio:.1%} "
                    f"warning_threshold={warning_threshold:.1%}"
                )
                lines.append(warning_message)
        for line in lines:
            self._log_info(line)