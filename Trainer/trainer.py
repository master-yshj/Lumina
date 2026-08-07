import torch
import pathlib
from math import isfinite

from Trainer.checkpoint import build_checkpoint, save_checkpoint

class Trainer:
    def __init__(self, task, optimizer, device, scheduler=None, logger=None):
        self.task = task
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.logger = logger

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
        detached_loss = outputs["loss"].item()
        new_dict = {"loss": detached_loss}
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
        detached_loss = outputs["loss"].item()
        new_dict = {"loss": detached_loss}
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
        total_loss = 0.0
        last_loss = None
        completed_batches = 0
        avg_loss = 0.0
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

            completed_batches += 1
            total_loss += loss
            last_loss = loss
            avg_loss = total_loss / completed_batches

            self._log_info(f"batch={completed_batches}/{batch_num}, loss={loss:.6f}, avg_loss={avg_loss:.6f}")

        if completed_batches == 0:
            raise ValueError(f"valid_dataloader是空的喵!")

        summary_dict = {
            "completed_batches": completed_batches,
            "last_loss": last_loss,
            "avg_loss": avg_loss,
        }
        self._log_info(f"batches={completed_batches}, avg_loss={avg_loss:.6f}")
        return summary_dict


    def fit(self, train_dataloader, total_iters, log_every_iters=100, accumulation_steps=1, valid_dataloader=None, do_validation=False, valid_every_iters=None, early_stop_patience=None, validation_rounds_before_save_ckpt=None, checkpoints_dir = None ):
        """
        fit方法负责不断从dataloader中取出数据，并调用train_step方法
        :param train_dataloader: 训练数据加载器
        :param total_iters: 需要完成的总迭代次数
        :param log_every_iters: 每隔多少个iter打印一次日志
        :param accumulation_steps: 梯度积累
        :param valid_dataloader: 验证数据加载器
        :param do_validation: 是否进行验证
        :param valid_every_iters: 每隔多少步验证一次
        :param early_stop_patience: 早停的patience
        :param validation_rounds_before_save_ckpt: 每验证多少轮保存一次检查点
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
                            "validation_rounds_before_save_ckpt": validation_rounds_before_save_ckpt,
                        }
        """
        if not type(total_iters) is int or total_iters <= 0:
            raise ValueError(f"total_iters必须为大于0的整数喵! 现在是: {total_iters}")
        if not type(log_every_iters) is int or log_every_iters <= 0:
            raise ValueError(f"log_every_iters必须为大于0的整数喵! 现在是: {log_every_iters}")
        if not type(accumulation_steps) is int or accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps必须为大于0的整数喵! 现在是: {accumulation_steps}")
        if not type(do_validation) is bool:
            raise ValueError(f"do_validation必须为bool类型的喵! 现在的type: {type(do_validation)}")
        if do_validation:
            if valid_dataloader is None:
                raise ValueError(f"valid_dataloader在do_validation时不能为空喵!")
            if not type(valid_every_iters) is int or valid_every_iters <= 0:
                raise ValueError(f"valid_every_iters必须为正整数喵! 现在是: {valid_every_iters}")
            if not type(early_stop_patience) is int or early_stop_patience <= 0:
                raise ValueError(f"early_stop_patience必须为正整数喵! 现在是: {early_stop_patience}")
        if checkpoints_dir is None and validation_rounds_before_save_ckpt is None:
            save_ckpt = False
        elif checkpoints_dir is None and validation_rounds_before_save_ckpt is not None:
                raise ValueError(f"checkpoints_dir 和 validation_rounds_before_save_ckpt 不能一个为空一个不为空喵!")
        elif checkpoints_dir is not None and validation_rounds_before_save_ckpt is None:
                raise ValueError(f"checkpoints_dir 和 validation_rounds_before_save_ckpt 不能一个为空一个不为空喵!")
        else:
            if not type(checkpoints_dir) in (str, pathlib.Path):
                raise ValueError(f"checkpoints_dir 必须为字符串类型或者pathlib.Path类型的喵! 现在的type: {type(checkpoints_dir)}")
            if isinstance(checkpoints_dir, str):
                if not checkpoints_dir.strip():
                    raise ValueError(f"当 checkpoints_dir 为字符串类型时不能为空喵!")
            checkpoints_dir = pathlib.Path(checkpoints_dir)
            if not checkpoints_dir.exists():
                raise ValueError(f"checkpoints_dir: {checkpoints_dir} 指向的路径不存在喵!")
            if not checkpoints_dir.is_dir():
                raise ValueError(f"checkpoints_dir: {checkpoints_dir} 指向的必须为目录喵!")
            if not type(validation_rounds_before_save_ckpt) is int or validation_rounds_before_save_ckpt <= 0:
                raise ValueError(f"validation_rounds_before_save_ckpt: {validation_rounds_before_save_ckpt} 必须为正整数喵!")
            save_ckpt = True

        total_loss = 0.0                            # 总损失
        last_loss = None                            # 上轮损失
        accumulated_batches = 0                     # 上次参数更新后，积累的batch数
        current_cycle_size = None                   # 当前进行梯度积累需要的batch数
        last_validation_summary = None              # 最近一次验证返回的内容
        completed_micro_steps = 0                   # 完成的micro batch数，与total_loss保持同步
        early_stopped = False                       # 是否早停提前结束

        while self.global_step < total_iters:
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

                self.micro_step += 1
                completed_micro_steps += 1
                accumulated_batches += 1
                total_loss += loss
                last_loss = loss
                avg_loss = total_loss / completed_micro_steps

                if should_update:
                    self.global_step += 1
                    accumulated_batches = 0

                    if self.global_step % log_every_iters == 0:
                        progress = self.global_step / total_iters * 100
                        self._log_info(f"iters: {self.global_step}/{total_iters} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}")

                    if do_validation and self.global_step % valid_every_iters == 0:
                        last_validation_summary = self.validate(valid_dataloader)
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
                                progress = self.global_step / total_iters * 100
                                self._log_info(f"checkpoint: best.pt, iters: {self.global_step}/{total_iters} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}")
                        else:
                            self.validation_rounds_without_improvement += 1
                        if self.validation_rounds_without_improvement >= early_stop_patience:
                            early_stopped = True
                            progress = self.global_step / total_iters * 100
                            self._log_info(f"early_stopped, iters: {self.global_step}/{total_iters} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}")

                        if save_ckpt and self.completed_validations % validation_rounds_before_save_ckpt == 0:
                            state_dict = build_checkpoint(
                                task=self.task,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                trainer=self,
                            )
                            last_checkpoint_path = checkpoints_dir / "last.pt"
                            save_checkpoint(state_dict, last_checkpoint_path)
                            progress = self.global_step / total_iters * 100
                            self._log_info(f"checkpoint: last.pt, iters: {self.global_step}/{total_iters} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}")

                if self.global_step >= total_iters:
                    break
                if early_stopped:
                    break

            if not batch_exist:
                raise ValueError(f"训练出现异常喵! dataloader为空喵! ")

        if save_ckpt:
            state_dict = build_checkpoint(
                task=self.task,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer=self,
            )
            last_checkpoint_path = checkpoints_dir / "last.pt"
            save_checkpoint(state_dict, last_checkpoint_path)
            progress = self.global_step / total_iters * 100
            self._log_info(f"checkpoint: last.pt, iters: {self.global_step}/{total_iters} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}")

        if not self.global_step % log_every_iters == 0:
            progress = self.global_step / total_iters * 100
            self._log_info(f"iters: {self.global_step}/{total_iters} ({progress:.2f}%), micro_step: {self.micro_step}, loss: {loss:.6f}, avg_loss: {avg_loss:.6f}")

        summary_dict = {
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
