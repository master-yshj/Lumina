import torch
import torch.nn as nn

from math import isfinite

class BertMLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        """
        Bert的MLM任务头
        :param hidden_size: 隐藏层维度
        :param vocab_size: 词表大小
        :param layer_norm_eps: 除以0的微小保护值
        """
        super(BertMLMHead, self).__init__()
        # 接收参数
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        # 校验参数
        if not type(self.hidden_size) is int or self.hidden_size <= 0:
            raise ValueError(f"hidden_size必须为严格正整数喵！现在是: {self.hidden_size}")
        if not type(self.vocab_size) is int or self.vocab_size <= 0:
            raise ValueError(f"vocab_size的值必须为严格正整数喵！现在是：{self.vocab_size}")
        if not type(self.layer_norm_eps) in (int, float) or self.layer_norm_eps <= 0 or not isfinite(self.layer_norm_eps):
            raise ValueError(f"layer_norm_eps的值必须为有限正数喵！现在是 {self.layer_norm_eps} ")

        # 构建内部层
        self.transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.GELU()
        self.vocab_projection = nn.Linear(self.hidden_size, self.vocab_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, hidden_states):
        """
        BertMLMHead的前向传播方法
        :param hidden_states: 三维浮点 Tensor，[B,S,H]
        :return: logits: 三维浮点 Tensor，[B,S,V]
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError(f"hidden_states的类型必须为Tensor喵！现在的type是：{type(hidden_states)}")
        if not hidden_states.is_floating_point():
            raise ValueError(f"hidden_states的类型必须为浮点Tensor喵！现在dtype是：{hidden_states.dtype}")
        if not hidden_states.dim() == 3:
            raise ValueError(f"hidden_states的类型必须为三维Tensor喵！")
        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        if not type(batch_size) is int or batch_size <= 0:
            raise ValueError(f"hidden_states的batch_size必须为正整数喵！")
        if not type(seq_length) is int or seq_length <= 0:
            raise ValueError(f"hidden_states的seq_length必须为正整数喵！")
        if not hidden_states.size(2) == self.hidden_size:
            raise ValueError(f"hidden_states的最后一维必须和hidden_size的大小一致喵！")
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.vocab_projection(hidden_states)
        return logits
