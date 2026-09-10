import torch
import pathlib
from math import isfinite

from Trainer.checkpoint import build_checkpoint, save_checkpoint
from device import get_memory_summary, reset_peak_memory_stats, get_memory_warning, get_oom_info

class Trainer:
    def __init__(self, task, optimizer, device, device_summary, train_metric_collection, valid_metric_collection, train_metrics_writer, valid_metrics_writer, scheduler=None, logger=None):
        self.task = task
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.device_summary = device_summary
        self.scheduler = scheduler
        self.logger = logger
        self.train_metric_collection = train_metric_collection
        self.valid_metric_collection = valid_metric_collection
        self.train_metrics_writer = train_metrics_writer
        self.valid_metrics_writer = valid_metrics_writer

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
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.to(self.device)
            else:
                new_batch[k] = v
        return new_batch

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

    def _log_memory_status(self):
        memory_summary = get_memory_summary(self.device_summary)
        self._log_info(f"memory summary: {memory_summary}")
        warning_summary = get_memory_warning(memory_summary)
        if warning_summary is None:
            pass
        else:
            backend = warning_summary["backend"]
            if backend == "mps":
                warning_check_available = warning_summary["warning_check_available"]
                is_warning = warning_summary["is_warning"]
                warning_threshold = warning_summary["warning_threshold"]
                usage_ratio = warning_summary["usage_ratio"]
                recommended_max_memory = warning_summary["recommended_max_memory"]
                driver_allocated_memory = warning_summary["driver_allocated_memory"]
                if warning_check_available:
                    if is_warning:
                        self._log_info(f"MemoryWarning: 当前显存占用过高了喵! 预警阈值: {warning_threshold} 占用比例: {usage_ratio} driver_allocated_memory: {driver_allocated_memory} recommended_max_memory: {recommended_max_memory}")
                    else:
                        pass
                else:
                    self._log_info(f"recommended_max_memory: {recommended_max_memory} 不能计算显存占用喵!")
            elif backend == "cuda":
                gpu_nums = warning_summary["gpu_nums"]
                if gpu_nums == 1:
                    device_index = warning_summary["device_index"]
                    warning_check_available = warning_summary["warning_check_available"]
                    is_warning = warning_summary["is_warning"]
                    warning_threshold = warning_summary["warning_threshold"]
                    usage_ratio = warning_summary["usage_ratio"]
                    max_reserved_memory = warning_summary["max_reserved_memory"]
                    total_memory = warning_summary["total_memory"]
                    if warning_check_available:
                        if is_warning:
                            self._log_info(f"MemoryWarning: 当前cuda: {device_index} 显存占用过高了喵! 预警阈值: {warning_threshold} 占用比例: {usage_ratio} 当前占用显存: {max_reserved_memory} 该显卡最大显存: {total_memory}")
                        else:
                            pass
                    else:
                        pass
                elif gpu_nums > 1:
                    for device_index in range(gpu_nums):
                        sub_summary = warning_summary["devices"][device_index]
                        warning_check_available = sub_summary["warning_check_available"]
                        is_warning = sub_summary["is_warning"]
                        warning_threshold = sub_summary["warning_threshold"]
                        usage_ratio = sub_summary["usage_ratio"]
                        max_reserved_memory = sub_summary["max_reserved_memory"]
                        total_memory = sub_summary["total_memory"]
                        if warning_check_available:
                            if is_warning:
                                self._log_info(
                                    f"MemoryWarning: 当前cuda: {device_index} 显存占用过高了喵! 预警阈值: {warning_threshold} 占用比例: {usage_ratio} 当前占用显存: {max_reserved_memory} 该显卡最大显存: {total_memory}")
                            else:
                                pass
                        else:
                            pass


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
        if not isinstance(batch, dict):
            raise ValueError(f"传入的batch必须为dict类型的喵! 现在type: {type(batch)}")
        if not type(should_zero_grad) is bool:
            raise ValueError(f"传入的should_zero_grad必须为bool类型的喵! 现在的type: {type(should_zero_grad)}")
        if not type(should_update) is bool:
            raise ValueError(f"传入的should_update必须为bool类型的喵! 现在的type: {type(should_update)}")
        if not type(loss_divisor) is int or loss_divisor <= 0:
            raise ValueError(f"传入的loss_divisor: {loss_divisor} 必须为正整数喵!")
        try:
            self.task.train()
            batch = self._transfer_batch_device(batch)
            if should_zero_grad:
                self.optimizer.zero_grad(set_to_none=True)
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
            backward_loss = original_loss / loss_divisor
            backward_loss.backward()
            if should_update:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        except Exception as error:
            oom_info_dict = get_oom_info(error,self.device_summary,"Train Step")
            self.optimizer.zero_grad(set_to_none=True)
            if oom_info_dict is None:
                raise
            else:
                self._log_info(f"OutOfMemory: {oom_info_dict}")
                raise RuntimeError(f"OutOfMemory: {oom_info_dict}") from error
        total_loss = outputs["loss"].item()
        new_dict = {"loss": total_loss}
        new_dict["losses"] = {}
        if not "losses" in outputs:
            raise ValueError(f"执行训练任务返回的dict必须包含 'losses' 这个键喵!")
        if not isinstance(outputs["losses"], dict):
            raise ValueError(f"执行训练任务返回的dict的losses的值必须为dict类型的喵! 现在是: {type(outputs['losses'])}")
        if not outputs["losses"]:
            raise ValueError(f"执行训练任务返回的dict的losses的值不能为空喵!")
        for loss_name, loss_info in outputs["losses"].items():
            if not isinstance(loss_info, dict):
                raise ValueError(f"outputs.losses.{loss_name} 的值必须为dict类型的喵!")
            if not loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 不能为空喵!")
            if not "raw" in loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 缺少了raw字段喵!")
            raw = loss_info["raw"]
            if not isinstance(raw, torch.Tensor) or not raw.dim() == 0 or not torch.isfinite(raw).item():
                raise ValueError(f"raw 必须为有限零维张量喵!")
            if not "weight" in loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 缺少了weight字段喵!")
            weight = loss_info["weight"]
            if not type(weight) in (float, int) or not isfinite(weight):
                raise ValueError(f"weight 必须为整型或浮点型喵!")
            if not "weighted" in loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 缺少了weighted字段喵!")
            weighted = loss_info["weighted"]
            if not isinstance(weighted, torch.Tensor) or not weighted.dim() == 0 or not torch.isfinite(weighted).item():
                raise ValueError(f"weighted 必须为有限零维张量喵!")
            raw = raw.item()
            weighted = weighted.item()
            new_info = {"raw": raw,
                        "weight": weight,
                        "weighted": weighted
                        }
            new_dict["losses"][loss_name] = new_info

        context = {
            "batch": batch,
            "outputs": outputs,
        }
        self.train_metric_collection.update(**context)

        return new_dict

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
        if not isinstance(batch, dict):
            raise ValueError(f"传入的batch必须为dict类型的喵! 现在type: {type(batch)}")
        self.task.eval()
        try:
            batch = self._transfer_batch_device(batch)
            with torch.no_grad():
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
        except Exception as error:
            oom_info_dict = get_oom_info(error,self.device_summary,"Validation")
            if oom_info_dict is None:
                raise
            else:
                self._log_info(f"OutOfMemory: {oom_info_dict}")
                raise RuntimeError(f"OutOfMemory: {oom_info_dict}") from error
        total_loss = outputs["loss"].item()
        new_dict = {"loss": total_loss}
        new_dict["losses"] = {}
        if not "losses" in outputs:
            raise ValueError(f"执行训练任务返回的dict必须包含 'losses' 这个键喵!")
        if not isinstance(outputs["losses"], dict):
            raise ValueError(f"执行训练任务返回的dict的losses的值必须为dict类型的喵! 现在是: {type(outputs['losses'])}")
        if not outputs["losses"]:
            raise ValueError(f"执行训练任务返回的dict的losses的值不能为空喵!")
        for loss_name, loss_info in outputs["losses"].items():
            if not isinstance(loss_info, dict):
                raise ValueError(f"outputs.losses.{loss_name} 的值必须为dict类型的喵!")
            if not loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 不能为空喵!")
            if not "raw" in loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 缺少了raw字段喵!")
            raw = loss_info["raw"]
            if not isinstance(raw, torch.Tensor) or not raw.dim() == 0 or not torch.isfinite(raw).item():
                raise ValueError(f"raw 必须为有限零维张量喵!")
            if not "weight" in loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 缺少了weight字段喵!")
            weight = loss_info["weight"]
            if not type(weight) in (float, int) or not isfinite(weight):
                raise ValueError(f"weight 必须为整型或浮点型喵!")
            if not "weighted" in loss_info:
                raise ValueError(f"outputs.losses.{loss_name} 缺少了weighted字段喵!")
            weighted = loss_info["weighted"]
            if not isinstance(weighted, torch.Tensor) or not weighted.dim() == 0 or not torch.isfinite(weighted).item():
                raise ValueError(f"weighted 必须为有限零维张量喵!")
            raw = raw.item()
            weighted = weighted.item()
            new_info = {"raw": raw,
                        "weight": weight,
                        "weighted": weighted
                        }
            new_dict["losses"][loss_name] = new_info

        context = {
            "batch": batch,
            "outputs": outputs,
        }
        self.valid_metric_collection.update(**context)

        return new_dict

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
        self.valid_metric_collection.reset()
        total_loss = 0.0
        last_loss = None
        completed_batches = 0
        avg_loss = 0.0
        losses_dict = {}
        batch_num = len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            output_dict = self.valid_step(batch=batch)
            if not isinstance(output_dict, dict):
                raise ValueError(f"output_dict的值应该为dict类型喵! 现在是type: {type(output_dict)}")
            if not "loss" in output_dict:
                raise ValueError(f"output_dict里面不包含loss的键喵!")
            loss = output_dict["loss"]
            if not type(loss) in (float, int) or not isfinite(loss):
                raise ValueError(f"loss必须为有限浮点数或整数喵! 请检查loss是否为nan喵! loss: {loss}")
            if not "losses" in output_dict:
                raise ValueError(f"output_dict里面不包含losses的键喵!")
            if not isinstance(output_dict["losses"], dict):
                raise ValueError(f"output_dict的losses必须为dict类型的喵!")
            if not output_dict["losses"]:
                raise ValueError(f"output_dict的losses不能为空喵!")

            completed_batches += 1
            total_loss += loss
            last_loss = loss
            avg_loss = total_loss / completed_batches
            if losses_dict and not set(output_dict["losses"].keys()) == set(losses_dict.keys()):
                raise ValueError(f"output_dict与losses_dict之间的字段不一致喵!")
            for loss_name, loss_info in output_dict["losses"].items():
                raw = loss_info["raw"]
                weighted = loss_info["weighted"]
                weight = loss_info["weight"]

                if loss_name not in losses_dict:
                    losses_dict[loss_name] = {
                        "raw": 0.0,
                        "weighted": 0.0,
                        "weight": weight,
                    }
                else:
                    if not losses_dict[loss_name]["weight"] == loss_info["weight"]:
                        raise ValueError(
                            f"{loss_name} 的权重出现变化了喵! 现在是: {loss_info['weight']} 应该是: {losses_dict[loss_name]['weight']}")
                losses_dict[loss_name]["raw"] += raw
                losses_dict[loss_name]["weighted"] += weighted

            self._log_info(f"batch={completed_batches}/{batch_num}, loss={loss:.6f}, avg_loss={avg_loss:.6f}")

        if completed_batches == 0:
            raise ValueError(f"valid_dataloader是空的喵!")

        avg_losses_dict = {}
        for loss_name, loss_info in losses_dict.items():
            avg_losses_dict[loss_name] = {
                'avg_raw': loss_info["raw"] / completed_batches,
                'avg_weighted_loss': loss_info["weighted"] / completed_batches,
                'weight': loss_info["weight"],
            }

        metric_output = self.valid_metric_collection.compute()

        summary_dict = {
            "completed_batches": completed_batches,
            "last_loss": last_loss,
            "avg_loss": avg_loss,
            "losses": avg_losses_dict,
            "valid_metrics": metric_output,
        }

        valid_metrics_row = {
            "global_step": self.global_step,
            "avg_loss": summary_dict["avg_loss"],
        }
        for loss_name, loss_info in summary_dict["losses"].items():
            avg_raw = loss_info["avg_raw"]
            avg_weighted_loss = loss_info["avg_weighted_loss"]
            loss_raw_fieldname = f"{loss_name}_avg_raw"
            loss_weighted_fieldname = f"{loss_name}_avg_weighted_loss"
            valid_metrics_row[loss_raw_fieldname] = avg_raw
            valid_metrics_row[loss_weighted_fieldname] = avg_weighted_loss
        for metric_key, metric_value in summary_dict["valid_metrics"].items():
            valid_metrics_row[metric_key] = metric_value
        self.valid_metrics_writer.write(valid_metrics_row)

        self.valid_metric_collection.reset()

        self._log_info(f"batches={completed_batches}, avg_loss={avg_loss:.6f}\n"
                       f"metrics:{metric_output}")
        return summary_dict


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
        if not type(total_steps) is int or total_steps <= 0:
            raise ValueError(f"total_steps必须为大于0的整数喵! 现在是: {total_steps}")
        if not type(log_every_steps) is int or log_every_steps <= 0:
            raise ValueError(f"log_every_steps必须为大于0的整数喵! 现在是: {log_every_steps}")
        if not type(accumulation_steps) is int or accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps必须为大于0的整数喵! 现在是: {accumulation_steps}")
        if not type(validation_enabled) is bool:
            raise ValueError(f"validation_enabled必须为bool类型的喵! 现在的type: {type(validation_enabled)}")
        if validation_enabled:
            if valid_dataloader is None:
                raise ValueError(f"valid_dataloader在validation_enabled时不能为空喵!")
            if not type(validation_every_steps) is int or validation_every_steps <= 0:
                raise ValueError(f"validation_every_steps必须为正整数喵! 现在是: {validation_every_steps}")
            if not type(early_stopping_patience) is int or early_stopping_patience <= 0:
                raise ValueError(f"early_stopping_patience必须为正整数喵! 现在是: {early_stopping_patience}")
        if checkpoints_dir is None and checkpoint_save_every_rounds is None:
            save_ckpt = False
        elif checkpoints_dir is None and checkpoint_save_every_rounds is not None:
                raise ValueError(f"checkpoints_dir 和 checkpoint_save_every_rounds 不能一个为空一个不为空喵!")
        elif checkpoints_dir is not None and checkpoint_save_every_rounds is None:
                raise ValueError(f"checkpoints_dir 和 checkpoint_save_every_rounds 不能一个为空一个不为空喵!")
        else:
            if not isinstance(checkpoints_dir, (str, pathlib.Path)):
                raise ValueError(f"checkpoints_dir 必须为字符串类型或者pathlib.Path类型的喵! 现在的type: {type(checkpoints_dir)}")
            if isinstance(checkpoints_dir, str):
                if not checkpoints_dir.strip():
                    raise ValueError(f"当 checkpoints_dir 为字符串类型时不能为空喵!")
            checkpoints_dir = pathlib.Path(checkpoints_dir)
            if not checkpoints_dir.exists():
                raise ValueError(f"checkpoints_dir: {checkpoints_dir} 指向的路径不存在喵!")
            if not checkpoints_dir.is_dir():
                raise ValueError(f"checkpoints_dir: {checkpoints_dir} 指向的必须为目录喵!")
            if not type(checkpoint_save_every_rounds) is int or checkpoint_save_every_rounds <= 0:
                raise ValueError(f"checkpoint_save_every_rounds: {checkpoint_save_every_rounds} 必须为正整数喵!")
            save_ckpt = True

        reset_peak_memory_stats(self.device_summary)
        self.train_metric_collection.reset()

        total_loss = 0.0                            # 总损失
        last_loss = None                            # 上轮损失
        accumulated_batches = 0                     # 上次参数更新后，积累的batch数
        current_cycle_size = None                   # 当前进行梯度积累需要的batch数
        last_validation_summary = None              # 最近一次验证返回的内容
        completed_micro_steps = 0                   # 完成的micro batch数，与total_loss保持同步
        early_stopped = False                       # 是否早停提前结束
        losses_dict = {}                            # losses的各项信息的字典
        last_train_metric_output = None             # 上次训练指标计算的输出
        interval_total_loss = 0.0                   # 某一训练区间内总损失
        interval_completed_micro_steps = 0          # 某一训练区间内完成的micro batch数
        interval_losses_dict = {}                   # 某一训练区间内losses的各项信息的字典
        interval_avg_loss = 0.0                     # 某一训练区间内的平均损失
        last_interval_avg_loss = None               # 某一训练区间内的上次训练的平均损失
        last_interval_losses_dict = None            # 某一训练区间内的上次训练损失字典

        try:
            first_batch = next(iter(train_dataloader))
        except StopIteration:
            raise ValueError(f"训练出现异常了喵! 现在的train_dataloader为空喵!")
        self._train_dry_run(first_batch)
        self._log_info(f"Dry Run 已完成，以下是 Dry Run 显存占用的相关信息")
        self._log_memory_status()
        while self.global_step < total_steps:
            batch_exist = False
            batch_num = len(train_dataloader)
            if early_stopped:
                break
            for idx, batch in enumerate(train_dataloader):
                # 在新累积周期的第一个 Batch 前清空上一周期梯度
                should_zero_grad = accumulated_batches == 0
                if accumulated_batches == 0:
                    # 获取未处理的总的batch, 便于进行后续min操作
                    remaining_batches = batch_num - idx
                    # 获取当前实际需要处理的batch的数量，因为最后进行处理的batch的长度可能不够accumulation_steps了，所以需要进行min操作，这就是为什么要获取remaining_batches
                    current_cycle_size = min(accumulation_steps, remaining_batches)
                # 这里是获取将要处理本次batch后的处理batch数，用于判断是否需要参数更新，而不是刚开始初始化参数的更新
                next_accumulated_batches = accumulated_batches + 1
                should_update = next_accumulated_batches == current_cycle_size
                batch_exist = True
                output_dict = self.train_step(batch=batch, should_zero_grad=should_zero_grad, should_update=should_update, loss_divisor=current_cycle_size)
                if not isinstance(output_dict, dict):
                    raise ValueError(f"output_dict的值应该为dict类型喵! 现在是type: {type(output_dict)}")
                if not "loss" in output_dict:
                    raise ValueError(f"output_dict里面不包含loss的键喵!")
                loss = output_dict["loss"]
                if not type(loss) in (float, int) or not isfinite(loss):
                    raise ValueError(f"loss必须为有限浮点数或整数喵! 请检查loss是否为nan喵! loss: {loss}")
                if not "losses" in output_dict:
                    raise ValueError(f"output_dict里面不包含losses的键喵!")
                if not isinstance(output_dict["losses"], dict):
                    raise ValueError(f"output_dict的losses必须为dict类型的喵!")
                if not output_dict["losses"]:
                    raise ValueError(f"output_dict的losses不能为空喵!")

                self.micro_step += 1
                completed_micro_steps += 1
                accumulated_batches += 1
                total_loss += loss
                last_loss = loss
                avg_loss = total_loss / completed_micro_steps
                interval_total_loss += loss
                interval_completed_micro_steps += 1
                interval_avg_loss = interval_total_loss / interval_completed_micro_steps
                if losses_dict and not set(output_dict["losses"].keys()) == set(losses_dict.keys()):
                    raise ValueError(f"output_dict与losses_dict之间的字段不一致喵!")
                for loss_name, loss_info in output_dict["losses"].items():
                    raw = loss_info["raw"]
                    weighted = loss_info["weighted"]
                    weight = loss_info["weight"]

                    if loss_name not in losses_dict:
                        losses_dict[loss_name] = {
                            "raw": 0.0,
                            "weighted": 0.0,
                            "weight": weight,
                        }
                    else:
                        if not losses_dict[loss_name]["weight"] == loss_info["weight"]:
                            raise ValueError(f"{loss_name} 的权重出现变化了喵! 现在是: {loss_info['weight']} 应该是: {losses_dict[loss_name]['weight']}")
                    losses_dict[loss_name]["raw"] += raw
                    losses_dict[loss_name]["weighted"] += weighted

                    if loss_name not in interval_losses_dict:
                        interval_losses_dict[loss_name] = {
                            "raw": 0.0,
                            "weighted": 0.0,
                            "weight": weight,
                        }
                    else:
                        if not interval_losses_dict[loss_name]["weight"] == loss_info["weight"]:
                            raise ValueError(f"{loss_name} 的权重出现变化了喵! 现在是: {loss_info['weight']} 应该是: {interval_losses_dict[loss_name]['weight']}")
                    interval_losses_dict[loss_name]["raw"] += raw
                    interval_losses_dict[loss_name]["weighted"] += weighted

                if should_update:
                    self.global_step += 1
                    accumulated_batches = 0

                    should_validate = validation_enabled and (self.global_step % validation_every_steps == 0 or self.global_step >= total_steps)

                    if self.global_step % log_every_steps == 0:
                        progress = self.global_step / total_steps * 100
                        current_lr = self._get_current_lr()
                        self._log_info(f"iters: {self.global_step}/{total_steps} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}, interval_avg_loss: {interval_avg_loss:.6f}, lr: {current_lr:.6f}")
                        self._log_memory_status()

                    if should_validate:
                        last_train_metric_output = self.train_metric_collection.compute()
                        last_validation_summary = self.validate(valid_dataloader)
                        train_metric_row = {
                            "global_step": self.global_step,
                            "avg_loss": avg_loss,
                            "interval_avg_loss": interval_avg_loss,
                            "lr": self._get_current_lr(),
                        }
                        for loss_name, loss_info in losses_dict.items():
                            loss_raw_fieldname = f"{loss_name}_avg_raw"
                            loss_weighted_fieldname = f"{loss_name}_avg_weighted_loss"
                            avg_raw = loss_info["raw"] / completed_micro_steps
                            avg_weighted = loss_info["weighted"] / completed_micro_steps
                            train_metric_row[loss_raw_fieldname] = avg_raw
                            train_metric_row[loss_weighted_fieldname] = avg_weighted
                        for loss_name, loss_info in interval_losses_dict.items():
                            loss_raw_fieldname = f"{loss_name}_interval_avg_raw"
                            loss_weighted_fieldname = f"{loss_name}_interval_avg_weighted_loss"
                            interval_avg_raw = loss_info["raw"] / interval_completed_micro_steps
                            interval_avg_weighted_loss = loss_info["weighted"] / interval_completed_micro_steps
                            train_metric_row[loss_raw_fieldname] = interval_avg_raw
                            train_metric_row[loss_weighted_fieldname] = interval_avg_weighted_loss
                        for metric_key, metric_value in last_train_metric_output.items():
                            train_metric_row[metric_key] = metric_value
                        self.train_metrics_writer.write(train_metric_row)
                        last_validation_summary["train_metrics"] = last_train_metric_output
                        self._log_info(f"train_metrics: {last_train_metric_output}\n"
                                       f"valid_metrics: {last_validation_summary['valid_metrics']}")
                        last_interval_avg_loss = interval_avg_loss
                        last_interval_losses_dict = interval_losses_dict
                        interval_total_loss = 0.0
                        avg_interval_losses_dict = {}
                        for loss_name, loss_info in last_interval_losses_dict.items():
                            avg_interval_losses_dict[loss_name] = {
                                'avg_raw': loss_info['raw'] / interval_completed_micro_steps,
                                'avg_weighted_loss': loss_info['weighted'] / interval_completed_micro_steps,
                                'weight': loss_info['weight'],
                            }
                        interval_completed_micro_steps = 0
                        interval_losses_dict = {}
                        interval_avg_loss = 0.0
                        self.train_metric_collection.reset()
                        self.completed_validations += 1
                        current_validation_loss = last_validation_summary["avg_loss"]
                        if self.best_validation_loss is None or current_validation_loss < self.best_validation_loss:
                            self.best_validation_loss = current_validation_loss
                            self.validation_rounds_without_improvement = 0
                            if save_ckpt:
                                state_dict = build_checkpoint(
                                    task=self.task,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    trainer=self,
                                )
                                best_checkpoint_path = checkpoints_dir / "best.pt"
                                save_checkpoint(state_dict, best_checkpoint_path)
                                progress = self.global_step / total_steps * 100
                                current_lr = self._get_current_lr()
                                self._log_info(f"checkpoint: best.pt, iters: {self.global_step}/{total_steps} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}, interval_avg_loss: {last_interval_avg_loss:.6f}, lr: {current_lr:.6f}")
                        else:
                            self.validation_rounds_without_improvement += 1
                        if self.validation_rounds_without_improvement >= early_stopping_patience:
                            early_stopped = True
                            progress = self.global_step / total_steps * 100
                            current_lr = self._get_current_lr()
                            self._log_info(f"early_stopped, iters: {self.global_step}/{total_steps} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}, interval_avg_loss: {last_interval_avg_loss:.6f}, lr: {current_lr:.6f}")

                        if save_ckpt and self.completed_validations % checkpoint_save_every_rounds == 0:
                            state_dict = build_checkpoint(
                                task=self.task,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                trainer=self,
                            )
                            last_checkpoint_path = checkpoints_dir / "last.pt"
                            save_checkpoint(state_dict, last_checkpoint_path)
                            progress = self.global_step / total_steps * 100
                            current_lr = self._get_current_lr()
                            self._log_info(f"checkpoint: last.pt, iters: {self.global_step}/{total_steps} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}, interval_avg_loss: {last_interval_avg_loss:.6f}, lr: {current_lr}")

                if self.global_step >= total_steps:
                    break
                if early_stopped:
                    break

            if not batch_exist:
                raise ValueError(f"训练出现异常喵! dataloader为空喵! ")

        if not validation_enabled:
            last_train_metric_output = self.train_metric_collection.compute()
            train_metric_row = {
                "global_step": self.global_step,
                "avg_loss": avg_loss,
                "interval_avg_loss": interval_avg_loss,
                "lr": self._get_current_lr(),
            }
            for loss_name, loss_info in losses_dict.items():
                loss_raw_fieldname = f"{loss_name}_avg_raw"
                loss_weighted_fieldname = f"{loss_name}_avg_weighted_loss"
                avg_raw = loss_info["raw"] / completed_micro_steps
                avg_weighted = loss_info["weighted"] / completed_micro_steps
                train_metric_row[loss_raw_fieldname] = avg_raw
                train_metric_row[loss_weighted_fieldname] = avg_weighted
            for loss_name, loss_info in interval_losses_dict.items():
                loss_raw_fieldname = f"{loss_name}_interval_avg_raw"
                loss_weighted_fieldname = f"{loss_name}_interval_avg_weighted_loss"
                interval_avg_raw = loss_info["raw"] / interval_completed_micro_steps
                interval_avg_weighted_loss = loss_info["weighted"] / interval_completed_micro_steps
                train_metric_row[loss_raw_fieldname] = interval_avg_raw
                train_metric_row[loss_weighted_fieldname] = interval_avg_weighted_loss
            for metric_key, metric_value in last_train_metric_output.items():
                train_metric_row[metric_key] = metric_value
            self.train_metrics_writer.write(train_metric_row)
            self._log_info(f"train_metrics: {last_train_metric_output}")
            last_interval_avg_loss = interval_avg_loss
            last_interval_losses_dict = interval_losses_dict
            avg_interval_losses_dict = {}
            for loss_name, loss_info in last_interval_losses_dict.items():
                avg_interval_losses_dict[loss_name] = {
                    'avg_raw': loss_info['raw'] / interval_completed_micro_steps,
                    'avg_weighted_loss': loss_info['weighted'] / interval_completed_micro_steps,
                    'weight': loss_info['weight'],
                }
            interval_total_loss = 0.0
            interval_completed_micro_steps = 0
            interval_losses_dict = {}
            interval_avg_loss = 0.0
            self.train_metric_collection.reset()

        if save_ckpt:
            state_dict = build_checkpoint(
                task=self.task,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer=self,
            )
            last_checkpoint_path = checkpoints_dir / "last.pt"
            save_checkpoint(state_dict, last_checkpoint_path)
            progress = self.global_step / total_steps * 100
            current_lr = self._get_current_lr()
            self._log_info(f"checkpoint: last.pt, iters: {self.global_step}/{total_steps} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}, interval_avg_loss: {last_interval_avg_loss:.6f}, lr: {current_lr:.6f}")

        if not self.global_step % log_every_steps == 0:
            progress = self.global_step / total_steps * 100
            current_lr = self._get_current_lr()
            self._log_info(f"iters: {self.global_step}/{total_steps} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}, interval_avg_loss: {last_interval_avg_loss:.6f}, lr: {current_lr:.6f}")
            self._log_memory_status()

        avg_losses_dict = {}
        for loss_name, loss_info in losses_dict.items():
            avg_losses_dict[loss_name] = {
                'avg_raw': loss_info["raw"] / completed_micro_steps,
                'avg_weighted_loss': loss_info["weighted"] / completed_micro_steps,
                'weight': loss_info["weight"],
            }

        self._log_memory_status()

        summary_dict = {
                            "completed_iters": self.global_step,
                            "global_step": self.global_step,
                            "micro_step": self.micro_step,
                            "avg_loss": avg_loss,
                            "last_loss": last_loss,
                            "losses": avg_losses_dict,
                            "last_interval_losses": avg_interval_losses_dict,
                            "completed_validations": self.completed_validations,
                            "last_validation_summary": last_validation_summary,
                            "best_validation_loss": self.best_validation_loss,
                            "validation_rounds_without_improvement": self.validation_rounds_without_improvement,
                            "early_stopped": early_stopped,
                            "train_metrics": last_train_metric_output,
                            "last_interval_avg_loss": last_interval_avg_loss,
                        }

        return summary_dict

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
        }
        return training_dict

    def load_training_progress(self, training_dict):
        """
        加载训练进度状态
        :param training_dict: 训练状态参数
        :return:
        """
        if not isinstance(training_dict, dict):
            raise ValueError(f"传入的 training_dict 必须为dict类型的喵! 现在的type: {type(training_dict)}")
        if not training_dict:
            raise ValueError(f"传入的 training_dict 不能为空喵!")
        if "global_step" not in training_dict:
            raise ValueError(f"传入的 training_dict 不包含 global_step 字段喵!")
        if "micro_step" not in training_dict:
            raise ValueError(f"传入的 training_dict 不包含 micro_step 字段喵!")
        if "completed_validations" not in training_dict:
            raise ValueError(f"传入的 training_dict 不包含 completed_validations 字段喵!")
        if "best_validation_loss" not in training_dict:
            raise ValueError(f"传入的 training_dict 不包含 best_validation_loss 字段喵!")
        if "validation_rounds_without_improvement" not in training_dict:
            raise ValueError(f"传入的 training_dict 不包含 validation_rounds_without_improvement 字段喵!")

        global_step = training_dict["global_step"]
        micro_step = training_dict["micro_step"]
        completed_validations = training_dict["completed_validations"]
        best_validation_loss = training_dict["best_validation_loss"]
        validation_rounds_without_improvement = training_dict["validation_rounds_without_improvement"]

        allowed_keys = {
            "global_step",
            "micro_step",
            "completed_validations",
            "best_validation_loss",
            "validation_rounds_without_improvement",
        }

        if not type(global_step) is int or global_step < 0:
            raise ValueError(f"global_step: {global_step} 必须为非负整数喵!")
        if not type(micro_step) is int or micro_step < 0:
            raise ValueError(f"micro_step: {micro_step} 必须为非负整数喵!")
        if not type(completed_validations) is int or completed_validations < 0:
            raise ValueError(f"completed_validations: {completed_validations} 必须为非负整数喵!")
        if best_validation_loss is not None:
            if not type(best_validation_loss) in (float, int) or not isfinite(best_validation_loss):
                raise ValueError(f"best_validation_loss: {best_validation_loss} 必须为None或者有限数值喵!")
        if not type(validation_rounds_without_improvement) is int or validation_rounds_without_improvement < 0:
            raise ValueError(f"validation_rounds_without_improvement: {validation_rounds_without_improvement} 必须为非负整数喵!")
        # 跨字段检查
        if micro_step < global_step:
            raise ValueError(f"micro_step: {micro_step} 不能小于 global_step: {global_step} 喵!")
        if validation_rounds_without_improvement > completed_validations:
            raise ValueError(f"completed_validations: {completed_validations} 不能小于 validation_rounds_without_improvement: {validation_rounds_without_improvement} 喵!")
        if completed_validations == 0 and best_validation_loss is not None:
            raise ValueError("completed_validations 为 0 时，best_validation_loss 必须为 None 喵!")
        if completed_validations > 0 and best_validation_loss is None:
            raise ValueError("completed_validations 大于 0 时，best_validation_loss 不能为 None 喵!")

        extra_keys = training_dict.keys() - allowed_keys

        if extra_keys:
            raise ValueError(f"training_dict 里面出现了未知的字段喵: {extra_keys}")

        self.global_step = global_step
        self.micro_step = micro_step
        self.completed_validations = completed_validations
        self.best_validation_loss = best_validation_loss
        self.validation_rounds_without_improvement = validation_rounds_without_improvement
