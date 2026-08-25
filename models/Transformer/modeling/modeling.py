import torch.nn as nn

from ..embedding.embedding import BertEmbedding
from ..modules.modules import Encoder
from ..head.heads import BertMLMHead, BertQAHead


class TinyBertModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size,
        padding_idx,
        num_heads,
        intermediate_size,
        num_layers,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        layer_norm_eps=1e-12,
    ):
        """
        小型Bert模型
        :param vocab_size: 词表大小
        :param hidden_size: 隐藏层维度
        :param max_position_embeddings: 最大序列长度，embedding模型允许输入的最大序列长度
        :param type_vocab_size: 句子类型数，bert中通常为2
        :param padding_idx: 填充词索引
        :param num_heads: 多头自注意力的头数
        :param intermediate_size: 中间层的维度
        :param num_layers: EncoderBlock的层数
        :param embedding_dropout: embedding层的dropout
        :param attention_dropout: attention层的dropout
        :param hidden_dropout: 隐藏层的dropout
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(TinyBertModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.padding_idx = padding_idx
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.embedding_dropout_prob = embedding_dropout
        self.attention_dropout_prob = attention_dropout
        self.hidden_dropout_prob = hidden_dropout
        self.layer_norm_eps = layer_norm_eps

        self.embeddings = BertEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            padding_idx=self.padding_idx,
            dropout=self.embedding_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.encoder = Encoder(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            attention_dropout=self.attention_dropout_prob,
            hidden_dropout=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        小型Bert的前向传播方法
        :param input_ids: 二维longTensor, [batch_size, seq_length]
        :param attention_mask: 二维0/1Tensor, [batch_size, seq_length] 或 None
        :param token_type_ids: 二维longTensor, [batch_size, seq_length] 或 None
        :return: hidden_states: 完成前向传播的三维浮点Tensor, [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        return hidden_states

class TinyBertForMaskedLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size,
        padding_idx,
        num_heads,
        intermediate_size,
        num_layers,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        layer_norm_eps=1e-12,
    ):
        """
        小型Bert模型与MLM任务头相结合
        :param vocab_size: 词表大小
        :param hidden_size: 隐藏层维度
        :param max_position_embeddings: 最大序列长度，embedding模型允许输入的最大序列长度
        :param type_vocab_size: 句子类型数，bert中通常为2
        :param padding_idx: 填充词索引
        :param num_heads: 多头自注意力的头数
        :param intermediate_size: 中间层的维度
        :param num_layers: EncoderBlock的层数
        :param embedding_dropout: embedding层的dropout
        :param attention_dropout: attention层的dropout
        :param hidden_dropout: 隐藏层的dropout
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(TinyBertForMaskedLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.padding_idx = padding_idx
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.embedding_dropout_prob = embedding_dropout
        self.attention_dropout_prob = attention_dropout
        self.hidden_dropout_prob = hidden_dropout
        self.layer_norm_eps = layer_norm_eps

        self.bert = TinyBertModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            padding_idx=self.padding_idx,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            embedding_dropout=self.embedding_dropout_prob,
            attention_dropout=self.attention_dropout_prob,
            hidden_dropout=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.mlm_head = BertMLMHead(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            layer_norm_eps=self.layer_norm_eps,
        )

        # embedding 和 vocab_projection共享参数
        self._tie_weights()

    def _tie_weights(self):
        """
        共享参数，为了保证输入输出语义一致，还能减少需要参与计算的参数量，节约算力
        :return: None
        """
        self.mlm_head.vocab_projection.weight = (
            self.bert.embeddings.word_embeddings.weight
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Bert与MLM头结合的模型的前向传播方法
        :param input_ids: 二维longTensor, [batch_size, seq_length]
        :param attention_mask: 二维0/1Tensor, [batch_size, seq_length] 或 None
        :param token_type_ids: 二维longTensor, [batch_size, seq_length] 或 None
        :return: logits: 三维浮点Tensor, [batch_size, seq_length, vocab_size]
        """
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.mlm_head(hidden_states=hidden_states)
        return logits

class TinyBertForQuestionAnswering(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            max_position_embeddings,
            type_vocab_size,
            padding_idx,
            num_heads,
            intermediate_size,
            num_layers,
            embedding_dropout=0.1,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            layer_norm_eps=1e-12,
    ):
        """
        小型Bert模型与QA任务头相结合
        :param vocab_size: 词表大小
        :param hidden_size: 隐藏层维度
        :param max_position_embeddings: 最大序列长度，embedding模型允许输入的最大序列长度
        :param type_vocab_size: 句子类型数，bert中通常为2
        :param padding_idx: 填充词索引
        :param num_heads: 多头自注意力的头数
        :param intermediate_size: 中间层的维度
        :param num_layers: EncoderBlock的层数
        :param embedding_dropout: embedding层的dropout
        :param attention_dropout: attention层的dropout
        :param hidden_dropout: 隐藏层的dropout
        :param layer_norm_eps: 除以0的微小值保护
        """
        super(TinyBertForQuestionAnswering, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.padding_idx = padding_idx
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.embedding_dropout_prob = embedding_dropout
        self.attention_dropout_prob = attention_dropout
        self.hidden_dropout_prob = hidden_dropout
        self.layer_norm_eps = layer_norm_eps

        self.bert = TinyBertModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            padding_idx=self.padding_idx,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            embedding_dropout=self.embedding_dropout_prob,
            attention_dropout=self.attention_dropout_prob,
            hidden_dropout=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.qa_head = BertQAHead(
            hidden_size=self.hidden_size,
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Bert与QA头结合的模型的前向传播方法
        :param input_ids: 二维longTensor, [batch_size, seq_length]
        :param attention_mask: 二维0/1Tensor, [batch_size, seq_length] 或 None
        :param token_type_ids: 二维longTensor, [batch_size, seq_length] 或 None
        :return: logits_dict: 三维浮点Tensor字典, [batch_size, seq_length]
        """
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits_dict = self.qa_head(hidden_states=hidden_states)
        return logits_dict