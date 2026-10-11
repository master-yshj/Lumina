import torch
import torch.nn as nn

from math import sqrt, isfinite


def _extend_attention_mask(attention_mask):
    """
    将二维的attention_mask拓展到四维
    :param attention_mask: 二维Tensor, [batch_size, seq_length]
    :return: extended_attention_mask: 四维Tensor, [batch_size, 1, 1, seq_length]
    """
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError(f"输入的attention_mask必须为Tensor张量类型的数据喵！现在type是: {type(attention_mask)}")
    if not attention_mask.dim() == 2:
        raise ValueError(f"输入的attention_mask必须为二维Tensor喵！现在维度是: {attention_mask.dim()}")
    batch_size = attention_mask.size(0)
    seq_length = attention_mask.size(1)
    if not type(batch_size) is int or batch_size <= 0:
        raise ValueError(f"batch_size必须为正整数喵！")
    if not type(seq_length) is int or seq_length <= 0:
        raise ValueError(f"seq_length必须为正整数喵！")
    valid_attention_mask = (attention_mask == 0) | (attention_mask == 1)
    if not torch.all(valid_attention_mask):
        raise ValueError(f"attention_mask里面的值只能为0或1喵！")
    valid_key_mask = attention_mask.to(dtype=torch.bool)
    extended_attention_mask = valid_key_mask[:, None, None, :]
    return extended_attention_mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout=0.1):
        """
        多头自注意力机制
        :param hidden_size: 隐藏层维度，
        :param num_heads: 多头注意力的头数，隐藏层维度等于多头注意力头数乘以每个头的维度（最后会有类似concat起来，实际是transpose和reshape）
        :param attention_dropout: int或float, [0,1)
        """
        super(MultiHeadSelfAttention, self).__init__()

        #接收参数
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_dropout_prob = attention_dropout

        # 校验参数
        if not type(hidden_size) is int or hidden_size <= 0:
            raise ValueError(f"hidden_size必须为严格正整数喵！现在是: {hidden_size}")
        if not type(num_heads) is int or num_heads <= 0:
            raise ValueError(f"num_heads必须为严格正整数喵！现在是: {num_heads}")
        if not hidden_size % num_heads == 0:
            raise ValueError(f"hidden_size必须能被num_heads整除喵！hidden_size: {hidden_size}, num_heads: {num_heads}")
        if not type(self.attention_dropout_prob) in (int, float) or not 0 <= self.attention_dropout_prob < 1 or not isfinite(self.attention_dropout_prob):
            raise ValueError(f"attention_dropout的值必须为[0,1)之间的整数或浮点数喵！，现在是 {self.attention_dropout_prob}")

        self.head_dim = hidden_size // num_heads
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

        # Q, K, V 投影层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        """
        多头自注意力机制的前向传播方法
        :param hidden_states: 三维Tensor, [batch_size, seq_length, hidden_size]
        :param attention_mask: 二维Tensor，来自tokenizer，[batch_size, seq_length]
        :return: output: 三维Tensor, 多头自注意力的最终计算结果, [batch_size, seq_length, hidden_size]
        """
        # 校验参数
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
            raise ValueError(f"hidden_state的最后一维必须和hidden_size的大小一致喵！")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError(f"attention_mask的类型必须为Tensor张量喵！现在的type: {type(attention_mask)}")
            if not attention_mask.dim() == 2:
                raise ValueError(f"attention_mask的维度必须为2喵！现在的维度为: {attention_mask.dim()}")
            if not attention_mask.shape == hidden_states.shape[:2]:
                raise ValueError(
                    f"attention_mask 的形状必须与 hidden_states 的前两维一致喵！"
                    f"attention_mask.shape: {attention_mask.shape}，"
                    f"hidden_states.shape[:2]: {hidden_states.shape[:2]}"
                )
            if not attention_mask.device == hidden_states.device:
                raise ValueError(f"attention_mask和hidden_states所处的设备不同喵！attention_mask.device: {attention_mask.device}, hidden_states.device: {hidden_states.device}")
            extended_attention_mask = _extend_attention_mask(attention_mask)

        # Q, K, V
        query = self.query(hidden_states)   # [batch_size, seq_length, hidden_size]
        key = self.key(hidden_states)       # [batch_size, seq_length, hidden_size]
        value = self.value(hidden_states)   # [batch_size, seq_length, hidden_size]

        query = self._split_heads(query, batch_size, seq_length)    # [batch_size, num_heads, seq_length, head_dim]
        key = self._split_heads(key, batch_size, seq_length)        # [batch_size, num_heads, seq_length, head_dim]
        value = self._split_heads(value, batch_size, seq_length)    # [batch_size, num_heads, seq_length, head_dim]

        # 注意力分数计算
        # query: [batch_size, num_heads, seq_length, head_dim]
        # key.transpose: [batch_size, num_heads, head_dim, seq_length]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, num_heads, seq_length, seq_length]
        attention_scores = attention_scores / sqrt(self.head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~extended_attention_mask,torch.finfo(attention_scores.dtype).min)   # ~为取反操作，不然会把那些有效token填充为最小有限值,取反后屏蔽 Padding Key，并填充为当前浮点类型的最小有限值
        attention_weights = torch.softmax(attention_scores, dim=-1)     # 最后一维通常是k的seq_length，只是自注意力机制的特殊性，Q和K的来源相同，所以注释里面统一为seq_length
        attention_weights = self.attention_dropout(attention_weights)
        # attention_weights: [batch_size, num_heads, seq_length, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim]
        context = torch.matmul(attention_weights, value) # [batch_size, num_heads, seq_length, head_dim]
        context = context.transpose(1, 2).contiguous()  # [batch_size, seq_length ,num_heads, head_dim]
        context = context.reshape(batch_size, seq_length, self.hidden_size) # [batch_size, seq_length, hidden_size]
        output = self.output(context)
        return output

    def _split_heads(self, tensor, batch_size, seq_length):
        """
        拆分多头方法
        :param tensor: 需要被拆分的张量Tensor, [batch_size, seq_length, hidden_size]
        :param batch_size: 批次大小, int
        :param seq_length: 序列长度, int
        :return: tensor: [batch_size, num_heads, seq_length, head_dim]
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"tensor的类型必须为Tensor喵！现在type是：{type(tensor)}")
        if not tensor.is_floating_point():
            raise ValueError(f"传入的tensor必须为浮点Tensor类型喵！现在dtype是: {tensor.dtype}")
        if not tensor.dim() == 3:
            raise ValueError(f"tensor的类型必须为三维Tensor喵！")
        if not type(batch_size) is int or batch_size <= 0:
            raise ValueError(f"batch_size必须为严格正整数喵！")
        if not type(seq_length) is int or seq_length <= 0:
            raise ValueError(f"seq_length必须为严格正整数喵！")

        # 拆分多头
        tensor = tensor.reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # [batch_size, seq_length, num_heads, head_dim]
        tensor = tensor.transpose(1, 2) # [batch_size, num_heads, seq_length, head_dim]
        return tensor

class AttentionOutput(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, layer_norm_eps=1e-12):
        """
        Attention输出头，
        :param hidden_size: hidden_size: 隐藏层维度
        :param dropout: 神经元随机丢弃概率
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(AttentionOutput, self).__init__()
        # 接收参数
        self.hidden_size = hidden_size
        self.dropout_prob = dropout
        self.layer_norm_eps = layer_norm_eps

        # 校验参数
        if not type(hidden_size) is int or hidden_size <= 0:
            raise ValueError(f"hidden_size必须为严格正整数喵！现在是: {hidden_size}")
        if not type(self.dropout_prob) in (int,float) or not 0 <= self.dropout_prob < 1 or not isfinite(self.dropout_prob):
            raise ValueError(f"dropout的值必须为[0,1)之间的整数或浮点数喵！，现在是 {self.dropout_prob}")
        if not type(self.layer_norm_eps) in (int, float) or self.layer_norm_eps <= 0 or not isfinite(self.layer_norm_eps):
            raise ValueError(f"layer_norm_eps的值必须为有限正数喵！现在是 {self.layer_norm_eps} ")

        # 构建内部层
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)

    def forward(self, attention_output, residual_input):
        """
        AttentionOutput的前向传播方法
        :param attention_output: 注意力机制的输出, Tensor, [batch_size, seq_length, hidden_size]
        :param residual_input: 残差输入, Tensor, [batch_size, seq_length, hidden_size]
        :return: output: 残差相加输出, Tensor, [batch_size, seq_length, hidden_size]
        """
        # 校验参数
        if not isinstance(attention_output, torch.Tensor):
            raise ValueError(f"attention_output必须为Tensor类型的喵！现在的type: {type(attention_output)}")
        if not attention_output.is_floating_point():
            raise ValueError(f"attention_output必须为浮点Tensor类型的喵！现在的dtype: {attention_output.dtype}")
        if not attention_output.dim() == 3:
            raise ValueError(f"attention_output必须为三维的喵！现在的维度为: {attention_output.dim()}")
        if not isinstance(residual_input, torch.Tensor):
            raise ValueError(f"residual_input必须为Tensor类型的喵！现在的type: {type(residual_input)}")
        if not residual_input.is_floating_point():
            raise ValueError(f"residual_input必须为浮点Tensor类型的喵！现在的dtype: {residual_input.dtype}")
        if not residual_input.dim() == 3:
            raise ValueError(f"residual_input必须为三维的喵！现在的维度为: {residual_input.dim()}")
        if not attention_output.device == residual_input.device:
            raise ValueError(f"attention_output的设备必须和residual_input的设备相同喵! attention_output.device: {attention_output.device}, residual_input.device: {residual_input.device}")
        if attention_output.dtype != residual_input.dtype:
            device_type = attention_output.device.type
            autocast_enabled = torch.is_autocast_enabled(device_type)
            allowed_dtypes = (torch.float32, torch.float16, torch.bfloat16)
            if not autocast_enabled or not attention_output.dtype in allowed_dtypes or not residual_input.dtype in allowed_dtypes:
                raise ValueError(
                    f"attention_output和residual_input的dtype必须一致喵！"
                    f"attention_output.dtype: {attention_output.dtype}, "
                    f"residual_input.dtype: {residual_input.dtype}"
                )
        if not attention_output.shape == residual_input.shape:
            raise ValueError(f"attention_output的形状必须和residual_input的形状相同喵! attention_output.shape: {attention_output.shape}, residual_input.shape: {residual_input.shape}")
        batch_size, seq_length, hidden_size = attention_output.size()
        if not hidden_size == self.hidden_size:
            raise ValueError(f"hidden_size不匹配喵，现在的hidden_size: {hidden_size}，合法的hidden_size: {self.hidden_size}")
        if not type(batch_size) is int or batch_size <= 0:
            raise ValueError(f"hidden_states的batch_size必须为正整数喵！")
        if not type(seq_length) is int or seq_length <= 0:
            raise ValueError(f"hidden_states的seq_length必须为正整数喵！")

        attention_output = self.dropout(attention_output)
        output = residual_input + attention_output
        output = self.layer_norm(output)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        """
        FeedForwardNetwork 前馈神经网络
        :param hidden_size: 隐藏层维度
        :param intermediate_size: 中间层维度
        :param dropout: 丢失神经元概率
        """
        super(FeedForwardNetwork, self).__init__()
        # 接收参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout

        # 校验参数
        if not type(hidden_size) is int or hidden_size <= 0:
            raise ValueError(f"hidden_size必须为严格正整数喵！现在是: {hidden_size}")
        if not type(intermediate_size) is int or intermediate_size <= 0:
            raise ValueError(f"intermediate_size必须为严格正整数喵！现在是: {intermediate_size}")
        if not type(self.dropout_prob) in (int,float) or not 0 <= self.dropout_prob < 1 or not isfinite(self.dropout_prob):
            raise ValueError(f"dropout的值必须为[0,1)之间的整数或浮点数喵！，现在是 {self.dropout_prob}")

        # 构建内部层
        self.dropout = nn.Dropout(dropout)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.output = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        """
        FeedForwardNetwork 前向传播方法
        :param hidden_states: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        :return: output: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        """
        # 校验参数
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
            raise ValueError(f"hidden_state的最后一维必须和hidden_size的大小一致喵！")

        output = self.intermediate(hidden_states)
        output = self.activation(output)
        output = self.output(output)
        output = self.dropout(output)
        return output

class FeedForwardOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps = 1e-12):
        """
        FFN前馈神经网络输出头
        :param hidden_size: 隐藏层维度
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(FeedForwardOutput, self).__init__()
        # 接收参数
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps

        # 校验参数
        if not type(hidden_size) is int or hidden_size <= 0:
            raise ValueError(f"hidden_size必须为严格正整数喵！现在是: {hidden_size}")
        if not type(self.layer_norm_eps) in (int, float) or self.layer_norm_eps <= 0 or not isfinite(self.layer_norm_eps):
            raise ValueError(f"layer_norm_eps的值必须为有限正数喵！现在是 {self.layer_norm_eps} ")

        # 构建内部层
        self.layer_norm = nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)

    def forward(self, ffn_output, residual_input):
        """
        FFN前馈神经网络输出头的前向传播方法
        :param ffn_output: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        :param residual_input: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        :return: output: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        """
        # 校验参数
        if not isinstance(ffn_output, torch.Tensor):
            raise ValueError(f"ffn_output必须为Tensor类型的喵！现在的type: {type(ffn_output)}")
        if not ffn_output.is_floating_point():
            raise ValueError(f"ffn_output必须为浮点Tensor类型的喵！现在的dtype: {ffn_output.dtype}")
        if not ffn_output.dim() == 3:
            raise ValueError(f"ffn_output必须为三维的喵！现在的维度为: {ffn_output.dim()}")
        if not isinstance(residual_input, torch.Tensor):
            raise ValueError(f"residual_input必须为Tensor类型的喵！现在的type: {type(residual_input)}")
        if not residual_input.is_floating_point():
            raise ValueError(f"residual_input必须为浮点Tensor类型的喵！现在的dtype: {residual_input.dtype}")
        if not residual_input.dim() == 3:
            raise ValueError(f"residual_input必须为三维的喵！现在的维度为: {residual_input.dim()}")
        if not ffn_output.device == residual_input.device:
            raise ValueError(f"ffn_output的设备必须和residual_input的设备相同喵! ffn_output.device: {ffn_output.device}, residual_input.device: {residual_input.device}")
        if ffn_output.dtype != residual_input.dtype:
            device_type = ffn_output.device.type
            autocast_enabled = torch.is_autocast_enabled(device_type)
            allowed_dtypes = (torch.float32, torch.float16, torch.bfloat16)
            if not autocast_enabled or not ffn_output.dtype in allowed_dtypes or not residual_input.dtype in allowed_dtypes:
                raise ValueError(
                    f"ffn_output和residual_input的dtype必须一致喵！"
                    f"ffn_output.dtype: {ffn_output.dtype}, "
                    f"residual_input.dtype: {residual_input.dtype}"
                )
        if not ffn_output.shape == residual_input.shape:
            raise ValueError(f"ffn_output的形状必须和residual_input的形状相同喵! ffn_output.shape: {ffn_output.shape}, residual_input.shape: {residual_input.shape}")
        batch_size, seq_length, hidden_size = ffn_output.size()
        if not hidden_size == self.hidden_size:
            raise ValueError(f"hidden_size不匹配喵，现在的hidden_size: {hidden_size}，合法的hidden_size: {self.hidden_size}")
        if not type(batch_size) is int or batch_size <= 0:
            raise ValueError(f"hidden_states的batch_size必须为正整数喵！")
        if not type(seq_length) is int or seq_length <= 0:
            raise ValueError(f"hidden_states的seq_length必须为正整数喵！")
        output = residual_input + ffn_output
        output = self.layer_norm(output)
        return output

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attention_dropout=0.1, hidden_dropout=0.1,layer_norm_eps=1e-12):
        """
        单个Encoder模块
        :param hidden_size: 隐藏层维度
        :param num_heads: 多头自注意力的头数
        :param intermediate_size: ffn中间层维度
        :param attention_dropout: attention层丢弃神经元的概率
        :param hidden_dropout: 隐藏层丢弃神经元的概率
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(EncoderBlock, self).__init__()
        # 接收参数
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout_prob = attention_dropout
        self.hidden_dropout_prob = hidden_dropout
        self.layer_norm_eps = layer_norm_eps

        # 校验参数
        if not type(hidden_size) is int or hidden_size <= 0:
            raise ValueError(f"hidden_size必须为严格正整数喵！现在是: {hidden_size}")
        if not type(num_heads) is int or num_heads <= 0:
            raise ValueError(f"num_heads必须为严格正整数喵！现在是: {num_heads}")
        if not hidden_size % num_heads == 0:
            raise ValueError(f"hidden_size必须能被num_heads整除喵！hidden_size: {hidden_size}, num_heads: {num_heads}")
        if not type(self.attention_dropout_prob) in (int, float) or not 0 <= self.attention_dropout_prob < 1 or not isfinite(self.attention_dropout_prob):
            raise ValueError(f"attention_dropout的值必须为[0,1)之间的整数或浮点数喵！，现在是 {self.attention_dropout_prob}")
        if not type(self.hidden_dropout_prob) in (int, float) or not 0 <= self.hidden_dropout_prob < 1 or not isfinite(self.hidden_dropout_prob):
            raise ValueError(f"hidden_dropout的值必须为[0,1)之间的整数或浮点数喵！，现在是 {self.hidden_dropout_prob}")
        if not type(intermediate_size) is int or intermediate_size <= 0:
            raise ValueError(f"intermediate_size必须为严格正整数喵！现在是: {intermediate_size}")
        if not type(self.layer_norm_eps) in (int, float) or self.layer_norm_eps <= 0 or not isfinite(self.layer_norm_eps):
            raise ValueError(f"layer_norm_eps的值必须为有限正数喵！现在是 {self.layer_norm_eps} ")

        # 构建中间层
        self.self_attention = MultiHeadSelfAttention(hidden_size=self.hidden_size, num_heads=self.num_heads, attention_dropout=self.attention_dropout_prob)
        self.attention_output = AttentionOutput(hidden_size=self.hidden_size, dropout=self.hidden_dropout_prob, layer_norm_eps=self.layer_norm_eps)
        self.feed_forward = FeedForwardNetwork(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size, dropout=self.hidden_dropout_prob)
        self.feed_forward_output = FeedForwardOutput(hidden_size=self.hidden_size, layer_norm_eps=self.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        """
        EncoderBlock的前向传播方法
        :param hidden_states: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        :param attention_mask: 二维0/1Tensor, [batch_size, seq_length] 或 None
        :return: output: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        """
        attention_output = self.self_attention(hidden_states=hidden_states, attention_mask=attention_mask)
        attention_hidden_states = self.attention_output(attention_output, hidden_states)
        ffn_output = self.feed_forward(attention_hidden_states)
        output = self.feed_forward_output(ffn_output, attention_hidden_states)
        return output

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, num_layers, attention_dropout=0.1, hidden_dropout=0.1,layer_norm_eps=1e-12):
        """
        将num_layers个EncoderBlock组合成最终的Encoder
        :param hidden_size: 隐藏层维度
        :param num_heads: 多头自注意力的头数
        :param intermediate_size: ffn中间层维度
        :param num_layers: EncoderBlock的个数
        :param attention_dropout: attention层丢弃神经元的概率
        :param hidden_dropout: 隐藏层丢弃神经元的概率
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(Encoder, self).__init__()
        # 接收参数
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout_prob = attention_dropout
        self.num_layers = num_layers
        self.hidden_dropout_prob = hidden_dropout
        self.layer_norm_eps = layer_norm_eps

        # 校验参数
        if not type(self.num_layers) is int or self.num_layers <= 0:
            raise ValueError(f"num_layers必须为严格正整数喵！num_layers: {self.num_layers}")

        # 构建参数独立的EncoderBlock
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_size=self.intermediate_size,
                    attention_dropout=self.attention_dropout_prob,
                    hidden_dropout=self.hidden_dropout_prob,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Encoder的前向传播方法
        :param hidden_states: 三维浮点Tensor, [batch_size, seq_length, hidden_size]
        :param attention_mask: 二维0/1Tensor, [batch_size, seq_length] 或 None
        :return: hidden_states: 返回经过Encoder的结果Tensor, [batch_size, seq_length, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        return hidden_states


class Decoder(nn.Module):
    pass
class Transformer(nn.Module):
    pass
