import torch

from math import isfinite

class Tester:

    def __init__(self, task, test_dataloader, device, metric_collection, metrics_writer, logger=None):
        self.task = task
        self.test_dataloader = test_dataloader
        self.device = torch.device(device)
        self.metric_collection = metric_collection
        self.metrics_writer = metrics_writer
        self.logger = logger

    def _transfer_batch_device(self, batch):
        """
        迁移batch的设备到指定设备上面
        :param batch: dict
        :return: batch
        """
        if not isinstance(batch, dict):
            raise ValueError(f"batch 必须为dict类型的喵! 现在的type: {type(batch)}")
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

    def evaluate(self):
        total_loss = 0.0
        last_loss = None
        completed_batches = 0
        losses_dict = {}
        batch_num = len(self.test_dataloader)
        self.metric_collection.reset()
        self.task.to(self.device)
        self.task.eval()
        with torch.no_grad():
            for idx, batch in enumerate(self.test_dataloader):
                batch = self._transfer_batch_device(batch)
                outputs = self.task(batch)
                if not isinstance(outputs, dict):
                    raise ValueError(f"执行测试任务的结果必须为dict类型的喵! 现在type: {type(outputs)}")
                if not "loss" in outputs:
                    raise ValueError(f"执行测试任务返回的dict必须包含 'loss' 这个键喵!")
                loss = outputs["loss"]
                if not isinstance(loss, torch.Tensor):
                    raise ValueError(f"loss必须为Tensor喵! 现在的type为: {type(loss)}")
                if not loss.dim() == 0:
                    raise ValueError(f"loss必须为标量型Tensor喵(即0维)! 现在的维度为: {loss.dim()}")
                if not torch.isfinite(loss).item():
                    raise ValueError(f"loss出现NaN值或无限值! 测试崩了喵!")
                completed_batches += 1
                loss = loss.item()
                total_loss += loss
                last_loss = loss
                avg_loss = total_loss / completed_batches
                if not "losses" in outputs:
                    raise ValueError(f"执行测试任务返回的dict必须包含 'losses' 这个键喵!")
                if not isinstance(outputs["losses"], dict):
                    raise ValueError(f"执行测试任务返回的dict的losses的值必须为dict类型的喵! 现在是: {type(outputs['losses'])}")
                if not outputs["losses"]:
                    raise ValueError(f"执行测试任务返回的dict的losses的值不能为空喵!")
                if losses_dict and not set(outputs["losses"].keys()) == set(losses_dict.keys()):
                    raise ValueError(f"output_dict与losses_dict之间的字段不一致喵!")
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
                    if not isinstance(weighted, torch.Tensor) or not weighted.dim() == 0 or not torch.isfinite(
                            weighted).item():
                        raise ValueError(f"weighted 必须为有限零维张量喵!")
                    raw = raw.item()
                    weighted = weighted.item()

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

                context = {
                    "batch": batch,
                    "outputs": outputs,
                }
                self.metric_collection.update(**context)

                self._log_info(f"batch={completed_batches}/{batch_num}, loss={loss:.6f}, avg_loss={avg_loss:.6f}")

            if completed_batches == 0:
                raise ValueError(f"test_dataloader是空的喵!")

            avg_losses_dict = {}
            for loss_name, loss_info in losses_dict.items():
                avg_losses_dict[loss_name] = {
                    'avg_raw': loss_info["raw"] / completed_batches,
                    'avg_weighted_loss': loss_info["weighted"] / completed_batches,
                    'weight': loss_info["weight"],
                }

            metric_output = self.metric_collection.compute()

            summary_dict = {
                "completed_batches": completed_batches,
                "last_loss": last_loss,
                "avg_loss": avg_loss,
                "losses": avg_losses_dict,
                "test_metrics": metric_output,
            }

            test_metrics_row = {
                "global_step": 1,
                "avg_loss": summary_dict["avg_loss"],
            }
            for loss_name, loss_info in summary_dict["losses"].items():
                avg_raw = loss_info["avg_raw"]
                avg_weighted_loss = loss_info["avg_weighted_loss"]
                loss_raw_fieldname = f"{loss_name}_avg_raw"
                loss_weighted_fieldname = f"{loss_name}_avg_weighted_loss"
                test_metrics_row[loss_raw_fieldname] = avg_raw
                test_metrics_row[loss_weighted_fieldname] = avg_weighted_loss
            for metric_key, metric_value in summary_dict["test_metrics"].items():
                test_metrics_row[metric_key] = metric_value

            self.metrics_writer.write(test_metrics_row)

            self.metric_collection.reset()

            self._log_info(f"batches={completed_batches}, avg_loss={avg_loss:.6f}\n"
                           f"metrics:{metric_output}")
            return summary_dict
