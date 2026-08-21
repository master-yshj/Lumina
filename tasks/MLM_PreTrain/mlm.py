import torch
import torch.nn as nn

class MLMTask(nn.Module):
    def __init__(self, model, loss_fn):
        """
        MLM任务
        :param model: 模型
        :param loss_fn: 损失函数
        """
        super(MLMTask, self).__init__()
        # 接收参数
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch):
        """
        MLM任务的前向传播方法
        :param batch: dict {
            "input_ids": input_ids, [batch_size, seq_length]
            "attention_mask": attention_mask, [batch_size, seq_length]
            "token_type_ids": token_type_ids, [batch_size, seq_length]
            "labels": labels, [batch_size, seq_length]
        }
        :return: output_dict, dict
        """
        if not isinstance(batch, dict):
            raise ValueError(f"batch必须为字典类型的喵! 现在的type为: {type(batch)}")
        if not "input_ids" in batch:
            raise ValueError(f"batch里面没有叫 'input_ids' 的字段喵! ")
        if not "attention_mask" in batch:
            raise ValueError(f"batch里面没有叫 'attention_mask' 的字段喵! ")
        if not "token_type_ids" in batch:
            raise ValueError(f"batch里面没有叫 'token_type_ids' 的字段喵! ")
        if not "labels" in batch:
            raise ValueError(f"batch里面没有叫 'labels' 的字段喵! ")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]

        output_dict = {}
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # [batch_size, seq_length, num_classes]
        output_dict["logits"] = logits
        if not logits.shape[:2] == labels.shape:
            raise ValueError(f"logits和labels的形状前两维度对不上喵！logits.shape: {logits.shape}, labels.shape: {labels.shape}")
        batch_size = logits.size(0)
        seq_length = logits.size(1)
        num_classes = logits.size(2)
        logits = logits.reshape(batch_size*seq_length, num_classes)
        labels = labels.reshape(batch_size*seq_length)
        loss = self.loss_fn(logits=logits, labels=labels)
        output_dict["loss"] = loss["loss"]
        output_dict["losses"] = loss["losses"]
        return output_dict
