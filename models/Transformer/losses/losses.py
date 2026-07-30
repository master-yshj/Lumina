import torch
import torch.nn as nn

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