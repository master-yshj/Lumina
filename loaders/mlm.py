import re
import json
import torch

from math import isfinite
from torch.utils.data import Dataset

def split_text_into_sentences(text):
    """
    将一篇较长的文章拆成句子
    :param text: 文章text, str
    :return: sentences: 拆分出来的句子,list
    """
    if not isinstance(text, str):
        raise ValueError(f"输入的text必须为字符串类型的喵！现在是: {type(text)}")
    if not text.strip():
        raise ValueError(f"输入的text不能为空喵！")
    parts = re.split(r"(?<=[。！？；!?])|\r?\n+", text)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences

def pack_sentences_into_token_chunks(sentences, tokenizer, max_seq_length):
    """
    把句子tokenize成token
    :param sentences: list, 句子列表
    :param tokenizer: 分词器
    :param max_seq_length: 最长序列长度, int
    :return: chunks，chunk分块后的列表
    """
    if not isinstance(sentences, list):
        raise ValueError(f"传入的sentences必须为list类型喵！现在是: {type(sentences)}")
    if len(sentences) == 0:
        raise ValueError(f"传入的sentences不能为空喵！")
    if not all(isinstance(sentence, str) for sentence in sentences):
        raise ValueError(f"sentences中的每个sentence都必须为字符串类型的喵！")
    if not all(sentence.strip() for sentence in sentences):
        raise ValueError(f"sentences中的每个sentence都必须不为空喵！")
    if not type(max_seq_length) is int or max_seq_length <= 2:
        raise ValueError(f"max_seq_length必须为大于2的整数喵！当前为: {max_seq_length}")
    content_capacity = max_seq_length - 2
    chunks = []
    current_chunk = []
    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(current_chunk) + len(sentence_tokens) <= content_capacity:
            current_chunk.extend(sentence_tokens)
        elif len(sentence_tokens) <= content_capacity:
            chunks.append(current_chunk)
            current_chunk = sentence_tokens
        else:
            remaining_capacity = content_capacity - len(current_chunk)
            current_chunk.extend(sentence_tokens[:remaining_capacity])
            sentence_tokens = sentence_tokens[remaining_capacity:]
            chunks.append(current_chunk)
            current_chunk = []
            while len(sentence_tokens) > content_capacity:
                current_chunk = sentence_tokens[:content_capacity]
                sentence_tokens = sentence_tokens[content_capacity:]
                chunks.append(current_chunk)
                current_chunk = []
            if len(sentence_tokens) > 0:
                current_chunk = sentence_tokens
    if len(current_chunk) > 0:
        chunks.append(current_chunk)
    return chunks

def build_mlm_sample(token_chunk, tokenizer):
    """
    创建用于mlm模型训练的样本
    :param token_chunk: 一个chunk里面的token列表
    :param tokenizer: 分词器
    :return: encoded_dict: {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
    }
    """

    if not isinstance(token_chunk, list):
        raise ValueError(f"token_chunk的类型必须为列表喵！现在为: {type(token_chunk)}")
    if len(token_chunk) == 0:
        raise ValueError(f"token_chunk不能为空喵！")
    if not all(isinstance(token, str) for token in token_chunk):
        raise ValueError(f"单个token必须为字符串类型的喵！")
    if not all(token.strip() for token in token_chunk):
        raise ValueError(f"单个token不能为空喵！")

    tokens = [tokenizer.cls_token] + token_chunk + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    encoded_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    return encoded_dict

class MLMDataset(Dataset):
    def __init__(self, data_file_path, tokenizer, max_seq_length, split_name, progress_interval, read_interval, logger=None):
        """
        MLM的Dataset构造
        :param data_file_path: 数据集文件路径
        :param tokenizer: 分词器
        :param max_seq_length: 序列最长长度
        :param split_name: 数据集名称 train, valid, test
        :param read_interval: 每读取到多少条记录输出一次日志
        :param progress_interval, 每处理多少条有效记录输出一次日志。
        :param logger:
        """
        # 接收参数
        self.data_file_path = data_file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split_name = split_name
        self.progress_interval = progress_interval
        self.read_interval = read_interval
        self.logger = logger
        self.samples = []

        # 校验参数
        if not self.data_file_path.is_file():
            raise ValueError(
                f"data_file_path: {self.data_file_path} 指向的不是一个文件喵！好好检查喵！")
        if self.data_file_path.stat().st_size == 0:
            raise ValueError(
                f"data_file_path: {self.data_file_path} 指向的文件是空的喵！")
        if not type(self.read_interval) is int or self.read_interval <= 0:
            raise ValueError(f"read_interval: {self.read_interval} 必须为正整数喵!")
        total_records = 0
        effective_records = 0
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    total_records += 1
                    effective_records += 1
                else:
                    total_records += 1
                if total_records % self.read_interval == 0:
                    self._log_info(f"已读取到 {total_records} 条记录，有效记录为 {effective_records} 条")
        if effective_records == 0:
            raise ValueError(f"data_file_path {self.data_file_path} 指向的文件是空的喵！")
        if not type(max_seq_length) is int or max_seq_length <= 2:
            raise ValueError(f"max_seq_length必须为大于2的整数喵！当前为: {max_seq_length}")
        if not isinstance(self.split_name, str) or self.split_name not in ["train", "valid", "test"]:
            raise ValueError(f"split_name: {self.split_name} 必须为 train, valid 或者 test 的字符串喵!")
        if not type(self.progress_interval) is int or self.progress_interval <= 0:
            raise ValueError(f"progress_interval: {self.progress_interval} 必须为正整数喵!")

        # 构造samples
        process_record = 0
        generated_samples = 0
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"数据文件 {self.data_file_path} 第 {line_number} 行不是合法JSON喵！"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"数据文件 {self.data_file_path} 第 {line_number} 行必须是JSON对象喵！"
                    )
                if "text" not in record:
                    raise ValueError(
                        f"数据文件 {self.data_file_path} 第 {line_number} 行缺少text字段喵！"
                    )
                text = record["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"数据文件 {self.data_file_path} 第 {line_number} 行的text必须是非空字符串喵！"
                    )
                sentences = split_text_into_sentences(text)
                token_chunks = pack_sentences_into_token_chunks(sentences=sentences, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
                for token_chunk in token_chunks:
                    sample = build_mlm_sample(
                        token_chunk=token_chunk,
                        tokenizer=self.tokenizer,
                    )
                    self.samples.append(sample)
                process_record += 1
                generated_samples = len(self.samples)
                if process_record % self.progress_interval == 0:
                    self._log_info(f"正在处理 {split_name} 数据集: 总记录数: {total_records} 有效记录数: {effective_records} 已处理 {process_record} 条记录，生成了 {generated_samples} 条sample")
            self._log_info(f"{split_name} 数据集已处理完毕喵~ 总记录数: {total_records} 有效记录数: {effective_records} 已处理 {process_record} 条记录，生成了 {generated_samples} 条sample")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _log_info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

class MLMCollator:
    def __init__(self, tokenizer, mlm_prob = 0.15):
        """
        MLM数据集整理器，用于DataLoader
        :param tokenizer: 分词器
        :param mlm_prob: 加掩码概率
        """
        # 接收参数
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

        if not type(self.mlm_prob) in (int,float) or not 0 < self.mlm_prob < 1 or not isfinite(self.mlm_prob):
            raise ValueError(f"mlm_prob的值必须为(0,1)之间的整数或浮点数喵！，现在是 {self.mlm_prob}")

    def __call__(self, samples):
        """
        将整个samples构建为一个Batch(根据batch_size分batch是后面DataLoader的任务)，并实现MLM数据集的整理
        15%挖空生成labels
        15% x 80% = 12% 的位置填充为[MASK]
        15% x (1 - 80%) x 50% = 1.5% 的位置填充为随机token
        15% x (1 - 80%) x (1 - 50%) = 1.5% 的位置什么都不填充
        :param samples: {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,}
        :return: mlm_dict {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        """
        # 检验参数
        if not isinstance(samples, list):
            raise ValueError(
                f"传入的samples必须为list类型喵！现在为: {type(samples)}"
            )
        if len(samples) == 0:
            raise ValueError("传入的samples不能为空喵！")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"传入的sample必须为dict类型的喵！现在为: {type(sample)}")
            if "input_ids" not in sample:
                raise ValueError(f"传入的sample里面缺少了一个叫 'input_ids' 的键！ ")
            if "attention_mask" not in sample:
                raise ValueError(f"传入的sample里面缺少了一个叫 'attention_mask' 的键！ ")
            if "token_type_ids" not in sample:
                raise ValueError(f"传入的sample里面缺少了一个叫 'token_type_ids' 的键！ ")

        # 动态补全padding，只补到最长input_ids
        batch_max_length = max(
            len(sample["input_ids"])
            for sample in samples
        )
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        for sample in samples:
            padding_length = batch_max_length - len(sample["input_ids"])
            batch_input_ids.append(sample["input_ids"] + ([self.tokenizer.pad_token_id] * padding_length))
            batch_attention_mask.append(sample["attention_mask"] + ([0] * padding_length))
            batch_token_type_ids.append(sample["token_type_ids"] + ([0] * padding_length))
        input_ids = torch.tensor(
            batch_input_ids,
            dtype=torch.long,
        )   # [batch_size, batch_max_length]
        attention_mask = torch.tensor(
            batch_attention_mask,
            dtype=torch.long,
        )   # [batch_size, batch_max_length]
        token_type_ids = torch.tensor(
            batch_token_type_ids,
            dtype=torch.long,
        )   # [batch_size, batch_max_length]
        # 构造标签
        labels = input_ids.clone()
        # 构造允许被监督的位置
        special_tokens_mask = (
                (input_ids == self.tokenizer.cls_token_id)
                | (input_ids == self.tokenizer.sep_token_id)
                | (input_ids == self.tokenizer.pad_token_id)
                | (attention_mask == 0)
        )
        # 创建每个位置的 MLM 抽样概率
        probability_matrix = torch.full(
            labels.shape,
            self.mlm_prob,
            dtype=torch.float,
        )
        probability_matrix.masked_fill_(
            special_tokens_mask,
            0.0,
        )
        # 二分类随机采样生成掩码，掩码的位置为True
        masked_positions = torch.bernoulli(
            probability_matrix
        ).to(dtype=torch.bool)
        # ~取反，把没有掩码的位置设为-100，因为不参与后续损失以及指标计算
        labels[~masked_positions] = -100
        # 有掩码的input_ids位置80%设置为[MASK]
        mask_matrix_prob = torch.full(
            labels.shape,
            0.8,
            dtype=torch.float,
        )
        mask_positions = torch.bernoulli(
            mask_matrix_prob
        ).to(dtype=torch.bool)
        replace_with_mask = (mask_positions & masked_positions)         # 被选中为掩码并且实际填充为MASK
        input_ids[replace_with_mask] = self.tokenizer.mask_token_id
        remaining_positions = (masked_positions & ~replace_with_mask)   # 被选中为掩码并且实际没有被填充为MASK，其中一半填充为随机token
        random_matrix_prob = torch.full(
            labels.shape,
            0.5,
            dtype=torch.float,
        )
        random_positions = torch.bernoulli(
            random_matrix_prob
        ).to(dtype=torch.bool)
        replace_with_random = (random_positions & remaining_positions)
        # 生成随机tokenID
        random_token_ids = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=labels.shape,
            dtype=torch.long,
        )
        input_ids[replace_with_random] = (
            random_token_ids[replace_with_random]
        )

        mlm_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

        return mlm_dict