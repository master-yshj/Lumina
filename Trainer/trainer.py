import torch

from math import isfinite

class Trainer:
    def __init__(self, task, optimizer, device):
        self.task = task
        self.optimizer = optimizer
        self.device = torch.device(device)

        self.task.to(self.device)

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

    def train_step(self, batch):
        """
        单步训练方法
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
        self.task.train()
        batch = self._transfer_batch_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.task(batch)
        if not isinstance(outputs, dict):
            raise ValueError(f"执行训练任务的结果必须为dict类型的喵! 现在type: {type(outputs)}")
        if not "loss" in outputs:
            raise ValueError(f"执行训练任务返回的dict必须包含 'loss' 这个键喵!")
        loss = outputs["loss"]
        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"loss必须为Tensor喵! 现在的type为: {type(loss)}")
        if not loss.dim() == 0:
            raise ValueError(f"loss必须为标量型Tensor喵(即0维)! 现在的维度为: {loss.dim()}")
        if not loss.isfinite():
            raise ValueError(f"loss出现NaN值或无限值! 训练崩了喵!")
        loss.backward()
        self.optimizer.step()
        detached_loss = outputs["loss"].item()
        new_dict = {"loss": detached_loss}
        return new_dict

    def fit(self, train_dataloader, total_iters, log_every_iters=100):
        """
        fit方法负责不断从dataloader中取出数据，并调用train_step方法
        :param train_dataloader: 训练数据加载器
        :param total_iters: 需要完成的总迭代次数
        :param log_every_iters: 每隔多少个iter打印一次日志
        :return:
        """
        if not type(total_iters) is int or total_iters <= 0:
            raise ValueError(f"total_iters必须为大于0的整数喵! 现在是: {total_iters}")
        if not type(log_every_iters) is int or log_every_iters <= 0:
            raise ValueError(f"log_every_iters必须为大于0的整数喵! 现在是: {log_every_iters}")

        completed_iters = 0
        total_loss = 0.0
        last_loss = None

        while completed_iters < total_iters:
            batch_exist = False
            for batch in train_dataloader:
                batch_exist = True
                output_dict = self.train_step(batch)
                if not isinstance(output_dict, dict):
                    raise ValueError(f"output_dict的值应该为dict类型喵! 现在是type: {type(output_dict)}")
                if not "loss" in output_dict:
                    raise ValueError(f"output_dict里面不包含loss的键喵!")
                loss = output_dict["loss"]
                if not type(loss) in (float, int) or not loss.isfinite():
                    raise ValueError(f"loss必须为有限浮点数或整数喵! 请检查loss是否为nan喵! loss: {loss}")

                total_loss += loss
                completed_iters += 1
                last_loss = loss
                avg_loss = total_loss / completed_iters

                if completed_iters % log_every_iters == 0:
                    print(f"iters: {completed_iters}, loss: {loss}, avg_loss: {avg_loss}")

                if completed_iters >= total_iters:
                    break

            if not batch_exist:
                raise ValueError(f"训练出现异常喵! dataloader为空喵! ")

        if not completed_iters % log_every_iters == 0:
            print(f"iters: {completed_iters}/{total_iters}, loss: {loss}, avg_loss: {avg_loss}")

        summary_dict = {"completed_iters": completed_iters,
                       "avg_loss": avg_loss,
                       "last_loss": last_loss,}

        return summary_dict