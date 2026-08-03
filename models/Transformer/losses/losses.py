import torch
import torch.nn as nn

from math import isfinite

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index = -100):
        """
        交叉熵损失
        """
        super().__init__()
        if type(ignore_index) is not int:
            raise TypeError(
                f"ignore_index 必须是 int 类型喵！当前类型为：{type(ignore_index)}"
            )
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
        )

    def forward(self, logits, labels):
        """
        交叉熵损失的前向传播
        :param logits: 分数，二维浮点Tensor，[batch_size x seq_length, num_classes]
        :param labels: 标签，一维longTensor，[batch_size x seq_length]
        :return: loss
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"logits 必须是 torch.Tensor 喵！当前类型为：{type(logits)}"
            )
        if not isinstance(labels, torch.Tensor):
            raise TypeError(
                f"labels 必须是 torch.Tensor 喵！当前类型为：{type(labels)}"
            )
        if logits.ndim != 2:
            raise ValueError(
                f"logits 必须是二维 Tensor [N, C] 喵！"
                f"当前形状为：{tuple(logits.shape)}"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"labels 必须是一维 Tensor [N] 喵！"
                f"当前形状为：{tuple(labels.shape)}"
            )
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits 和 labels 的样本数量必须一致喵！"
                f"当前分别为：{logits.shape[0]} 和 {labels.shape[0]}"
            )
        if not logits.is_floating_point():
            raise TypeError(
                f"logits 必须是浮点 Tensor 喵！当前 dtype 为：{logits.dtype}"
            )
        if labels.dtype != torch.long:
            raise TypeError(
                f"labels 必须是 torch.long 喵！当前 dtype 为：{labels.dtype}"
            )
        if logits.device != labels.device:
            raise ValueError(
                f"logits 和 labels 必须位于同一设备喵！"
                f"当前分别为：{logits.device} 和 {labels.device}"
            )
        loss = self.loss_fn(logits, labels)
        return loss

class WeightedLoss(nn.Module):
    def __init__(self, loss_list):
        """
        加权损失计算
        :param loss_list: list of dict: {
            name: str
            loss_fn: 实例化后的loss(nn.Module)
            weight: int or float
        }
        """
        super().__init__()
        # 接收参数
        self.loss_list = loss_list
        self.loss_fns = nn.ModuleDict()
        self.weights = {}
        # 校验参数
        if not isinstance(self.loss_list, list):
            raise TypeError(f"loss_list的值必须为list类型的喵! 现在是:{type(self.loss_list)}")
        if len(self.loss_list) == 0:
            raise ValueError(f"loss_list不能为空喵!")
        for idx, loss_dict in enumerate(self.loss_list):
            if not isinstance(loss_dict, dict):
                raise TypeError(f"loss_list[{idx}] 的值必须为dict类型的喵! 现在是:{type(loss_dict)}")
            if not "name" in loss_dict:
                raise ValueError(f"loss_list[{idx}] 中缺少了 'name' 这一字段喵!")
            if not "loss_fn" in loss_dict:
                raise ValueError(f"loss_list[{idx}] 中缺少了 'loss_fn' 这一字段喵!")
            if not "weight" in loss_dict:
                raise ValueError(f"loss_list[{idx}] 中缺少了 'weight' 这一字段喵!")
            name = loss_dict["name"]
            loss_fn = loss_dict["loss_fn"]
            weight = loss_dict["weight"]
            if not isinstance(name, str):
                raise ValueError(f"loss_list[{idx}].name: {name} 必须为字符串类型的喵! 现在的type:{type(name)}")
            if not name.strip():
                raise ValueError(f"loss_list[{idx}].name 不能为空喵!")
            if not isinstance(loss_fn, nn.Module):
                raise ValueError(f"loss_list[{idx}].loss_fn: {loss_fn} 必须为 nn.Module 类型的喵!")
            if not type(weight) in (int, float) or weight < 0 or not isfinite(weight):
                raise ValueError(f"loss_list[{idx}].weight: {weight} 必须为非负有限数值喵!")
            self.loss_fns[name] = loss_fn
            self.weights[name] = weight

    def forward(self, logits, labels):
        total_loss = 0.0
        for name, loss_fn in self.loss_fns.items():
            weight = self.weights[name]
            loss = loss_fn(logits=logits, labels=labels)
            total_loss += loss * weight
        return total_loss
