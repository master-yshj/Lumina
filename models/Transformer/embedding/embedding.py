import torch
import torch.nn as nn

from math import isfinite

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int, type_vocab_size: int, padding_idx: int, dropout=0.1, layer_norm_eps=1e-12):
        """
        BertEmbedding 嵌入层的初始化
        :param vocab_size: 词表大小
        :param hidden_size: 隐藏维度，每一个token映射后的隐藏空间维度
        :param max_position_embeddings: 最大序列长度，embedding模型允许输入的最大序列长度
        :param type_vocab_size: 句子类型数，bert中通常为2
        :param padding_idx: 填充词索引
        :param dropout: 神经元随机丢弃概率
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(BertEmbedding, self).__init__()
        # 接收参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.padding_idx = padding_idx
        self.dropout_prob = dropout
        self.layer_norm_eps = layer_norm_eps

        # 参数校验
        if not type(self.vocab_size) is int or self.vocab_size <= 0:
            raise ValueError(f"vocab_size的值必须为严格正整数喵！现在是：{self.vocab_size}")
        if not type(self.hidden_size) is int or self.hidden_size <= 0:
            raise ValueError(f"hidden_size的值必须为严格正整数喵！现在是：{self.hidden_size}")
        if not type(self.max_position_embeddings) is int or self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings的值必须为严格正整数喵！现在是：{self.max_position_embeddings}")
        if not type(self.type_vocab_size) is int or self.type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size的值必须为严格正整数喵！现在是：{self.type_vocab_size}")
        if not type(self.padding_idx) is int or not 0 <= self.padding_idx < self.vocab_size :
            raise ValueError(f"padding_idx的值必须为正整数喵！并且要小于vocab_size喵！现在的padding_idx: {self.padding_idx}, vocab_size: {self.vocab_size}")
        if not type(self.dropout_prob) in (int,float) or not 0 <= self.dropout_prob < 1 or not isfinite(self.dropout_prob):
            raise ValueError(f"dropout的值必须为[0,1)之间的整数或浮点数喵！，现在是 {self.dropout_prob}")
        if not type(self.layer_norm_eps) in (int, float) or self.layer_norm_eps <= 0 or not isfinite(self.layer_norm_eps):
            raise ValueError(f"layer_norm_eps的值必须为有限正数喵！现在是 {self.layer_norm_eps} ")

        # 初始化内部层
        self.word_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(num_embeddings=self.max_position_embeddings, embedding_dim=self.hidden_size)
        self.type_embeddings = nn.Embedding(num_embeddings=self.type_vocab_size, embedding_dim=self.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        embedding层的前向传播
        :param input_ids: [batch_size, seq_length]
        :param token_type_ids: : [batch_size, seq_length]
        :return: embedding: [batch_size, seq_length, hidden_size]，embedding的最终结果
        """
        # 参数校验
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(f"input_ids的类型必须是Tensor喵！")
        if not input_ids.dtype == torch.long:
            raise ValueError(f"input_ids的类型必须是Tensor.int64喵！")
        if token_type_ids is not None and not isinstance(token_type_ids, torch.Tensor):
            raise ValueError(f"token_type_ids的类型必须是Tensor或者None喵！")
        if not input_ids.dim() == 2:
            raise ValueError(f"input_ids必须为二维张量喵！input_ids.dim: {input_ids.dim()}")
        if token_type_ids is not None:
            if not input_ids.shape == token_type_ids.shape:
                raise ValueError(f"input_ids和token_type_ids的张量形状必须一样喵！input_ids.shape: {input_ids.shape}, token_type_ids: {token_type_ids.shape}")
        else:
            token_type_ids = torch.zeros_like(input_ids)
        if not token_type_ids.dtype == torch.long:
            raise ValueError(f"token_type_ids的类型必须是Tensor.int64喵！")
        if input_ids.device != token_type_ids.device:
            raise ValueError(f"input_ids和token_type_ids必须位于同一设备喵！"
                             f"input_ids.device: {input_ids.device}, "
                             f"token_type_ids.device: {token_type_ids.device}"
            )
        valid_input_ids = ((input_ids >= 0) & (input_ids < self.vocab_size))    #合法张量，使用torch.all对里面每个张量进行判断，由0和1的布尔值组成
        valid_token_type_ids = (token_type_ids >= 0) & (token_type_ids < self.type_vocab_size)  #合法张量，使用torch.all对里面每个张量进行判断，由0和1的布尔值组成
        if not torch.all(valid_input_ids):
            raise ValueError(f"input_ids里面所有的值都必须在词表中喵！有id越界了！")
        if not torch.all(valid_token_type_ids):
            raise ValueError(f"token_type_ids里面所有的值必须在规定范围内喵！有id越界了！请检查句子类型！")
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        if batch_size <= 0:
            raise ValueError(f"请检查input_ids的第0维！batch_size必须为正数喵！")
        if seq_length <= 0:
            raise ValueError(f"请检查input_ids的第1维！seq_length必须为正数喵！")
        if seq_length > self.max_position_embeddings:
            raise ValueError(f"输入的input_ids的seq_length必须小于max_position_embeddings: {self.max_position_embeddings} 喵！")

        # 生成position_ids 的tensor张量序列
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_length]

        # 三种embedding
        word_embeddings = self.word_embeddings(input_ids)               # [batch_size, seq_length, hidden_size]
        position_embeddings = self.position_embeddings(position_ids)    # [batch_size, seq_length, hidden_size]
        type_embeddings = self.type_embeddings(token_type_ids)          # [batch_size, seq_length, hidden_size]

        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings