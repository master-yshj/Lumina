import json
import torch
import pathlib

from copy import deepcopy
from torch.utils.data import Dataset

def build_qa_samples(tokenizer, data_dict):
    """
    创建用于qa模型训练的样本
    :param tokenizer: 分词器
    :param data_dict: jsonl文件的一行字典
    :return:
    """
    if not isinstance(data_dict, dict):
        raise ValueError(f"data_dict 必须为dict类型的喵! 现在的type: {type(data_dict)}")
    if not data_dict:
        raise ValueError(f"data_dict 不能为空喵!")
    if not "id" in data_dict:
        raise ValueError(f"data_dict 必须包含id这个字段喵!")
    data_id = data_dict["id"]
    if not "question" in data_dict:
        raise ValueError(f"data_dict 必须包含question这个字段喵!")
    question = data_dict["question"]
    if not "answers" in data_dict:
        raise ValueError(f"data_dict 必须包含answers这个字段喵!")
    answers = data_dict["answers"]
    if not "context" in data_dict:
        raise ValueError(f"data_dict 必须包含context这个字段喵!")
    context = data_dict["context"]
    encoded_dict = tokenizer.encode_with_offsets(text=question, text_pair=context, add_special_tokens=True, truncation=False, padding=False)
    input_ids = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]
    token_type_ids = encoded_dict["token_type_ids"]
    offset_mapping = encoded_dict["offset_mapping"]
    samples = []
    if not isinstance(answers, list):
        raise ValueError(f"answers 必须为list类型的喵! 现在的type: {type(answers)}")
    if not answers:
        raise ValueError(f"answers 不能为空喵!")
    for idx, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise ValueError(f"answer 必须为dict类型的喵! 现在的type: {type(answer)}")
        if not answer:
            raise ValueError(f"answer 不能为空喵!")
        text = answer["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"text 必须为非空字符串喵!")
        answer_start = answer["answer_start"]
        if not type(answer_start) is int or answer_start < 0:
            raise ValueError(f"answer_start: {answer_start} 必须为非负整数喵!")
        if not answer_start + len(text) <= len(context):
            raise ValueError(f"answer_end 已经超出了context喵! 越界了喵!")
        answer_end = answer_start + len(text)
        if not context[answer_start:answer_end] == text:
            raise ValueError(f"根据开始和结束位置取得的text和原文不匹配喵!")
        start_position = None
        end_position = None
        for sub_idx, (token_type_id, offset) in enumerate(zip(token_type_ids, offset_mapping)):
            token_start, token_end = offset
            if token_type_id == 1 and token_start <= answer_start < token_end:
                start_position = sub_idx
            if token_type_id == 1 and token_start < answer_end <= token_end:
                end_position = sub_idx
        if start_position is None or end_position is None:
            raise ValueError(f"答案没有落在完整编码的text中喵!")
        mapped_start = offset_mapping[start_position][0]
        mapped_end = offset_mapping[end_position][1]
        reconstructed_answer = context[mapped_start:mapped_end]
        if not reconstructed_answer == text:
            raise ValueError(f"answer的起始位置到token序列的映射出现错误了喵!")
        feature_dict = {
            "id": data_id,
            "question": question,
            "context": context,
            "answers": answers,
            "answer_index": idx,
            "answer_text": text,
            "answer_start": answer_start,
            "answer_end": answer_end,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "offset_mapping": offset_mapping,
            "start_position": start_position,
            "end_position": end_position,
        }
        samples.append(feature_dict)
    return samples

class QADataset(Dataset):
    def __init__(self, data_file_path, tokenizer, split_name, progress_interval, read_interval, logger=None):
        """
        QA的Dataset构造
        :param data_file_path: 数据集文件路径
        :param tokenizer: 分词器
        :param split_name: 数据集名称 train, valid, test
        :param read_interval: 每读取到多少条记录输出一次日志
        :param progress_interval, 每处理多少条有效记录输出一次日志。
        :param logger:
        """
        # 接收参数
        self.data_file_path = data_file_path
        self.tokenizer = tokenizer
        self.split_name = split_name
        self.progress_interval = progress_interval
        self.read_interval = read_interval
        self.logger = logger
        self.samples = []

        # 校验参数
        if not self.data_file_path.is_file():
            raise ValueError(f"data_file_path: {self.data_file_path} 指向的不是一个文件喵！好好检查喵！")
        if self.data_file_path.stat().st_size == 0:
            raise ValueError(f"data_file_path: {self.data_file_path} 指向的文件是空的喵！")
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
                    raise ValueError(f"数据文件 {self.data_file_path} 第 {line_number} 行不是合法JSON喵！") from error
                if not isinstance(record, dict):
                    raise ValueError(f"数据文件 {self.data_file_path} 第 {line_number} 行必须是JSON对象喵！")
                feature_list = build_qa_samples(tokenizer, record)
                self.samples.extend(feature_list)
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

class QACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        if not isinstance(features, list):
            raise ValueError(f"features 必须为list类型的喵! 现在的type: {type(features)}")
        if not features:
            raise ValueError(f"features 不能为空喵!")

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        start_position_list = []
        end_position_list = []

        id_list = []
        question_list = []
        context_list = []
        answers_list = []
        answer_index_list = []
        answer_text_list = []
        answer_start_list = []
        answer_end_list = []
        offset_mapping_list = []
        batch_max_length = 0

        for feature in features:
            if not isinstance(feature, dict):
                raise ValueError(f"feature 必须为dict类型的喵! 现在的type: {type(feature)}")
            if not "input_ids" in feature:
                raise ValueError(f"feature 中必须包含 input_ids 字段喵!")
            if not "attention_mask" in feature:
                raise ValueError(f"features 中必须包含 attention_mask 字段喵!")
            if not "token_type_ids" in feature:
                raise ValueError(f"feature 中必须包含 token_type_ids 字段喵!")
            if not "start_position" in feature:
                raise ValueError(f"features 中必须包含 start_position 字段喵!")
            if not "end_position" in feature:
                raise ValueError(f"feature 中必须包含 end_position 字段喵!")
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            token_type_ids = feature["token_type_ids"]
            start_position = feature["start_position"]
            end_position = feature["end_position"]
            if not len(input_ids) == len(attention_mask) == len(token_type_ids):
                raise ValueError(f"input_ids, attention_mask, token_type_ids 的长度必须一致喵!\n"
                                 f"len(input_ids): {len(input_ids)}\n"
                                 f"len(attention_mask): {len(attention_mask)}\n"
                                 f"len(token_type_ids): {len(token_type_ids)}\n")
            if not 0 <= start_position <= end_position < len(input_ids):
                raise ValueError(f"当前feature答案始末位置不能用于Collator构造batch数据喵!")

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)

            id_list.append(feature["id"])
            question_list.append(feature["question"])
            context_list.append(feature["context"])
            answers_list.append(feature["answers"])
            answer_index_list.append(feature["answer_index"])
            answer_text_list.append(feature["answer_text"])
            answer_start_list.append(feature["answer_start"])
            answer_end_list.append(feature["answer_end"])
            offset_mapping_list.append(feature["offset_mapping"])

            start_position_list.append(start_position)
            end_position_list.append(end_position)

            feature_length = len(input_ids)
            if feature_length > batch_max_length:
                batch_max_length = feature_length

        copy_input_ids_list = deepcopy(input_ids_list)
        copy_attention_mask_list = deepcopy(attention_mask_list)
        copy_token_type_ids_list = deepcopy(token_type_ids_list)

        for idx, (input_ids, attention_mask, token_type_ids) in enumerate(zip(copy_input_ids_list, copy_attention_mask_list, copy_token_type_ids_list)):
            if len(input_ids) < batch_max_length:
                copy_input_ids_list[idx].extend([self.tokenizer.pad_token_id] * (batch_max_length - len(input_ids)))
                copy_attention_mask_list[idx].extend([0] * (batch_max_length - len(attention_mask)))
                copy_token_type_ids_list[idx].extend([0] * (batch_max_length - len(token_type_ids)))

        input_ids_tensor = torch.tensor(copy_input_ids_list, dtype=torch.long)
        attention_mask_tensor = torch.tensor(copy_attention_mask_list, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(copy_token_type_ids_list, dtype=torch.long)
        start_position_tensor = torch.tensor(start_position_list, dtype=torch.long)
        end_position_tensor = torch.tensor(end_position_list, dtype=torch.long)

        batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "token_type_ids": token_type_ids_tensor,
            "start_positions": start_position_tensor,
            "end_positions": end_position_tensor,
            "id_list": id_list,
            "context_list": context_list,
            "answers_list": answers_list,
            "answer_text_list": answer_text_list,
            "answer_start_list": answer_start_list,
            "answer_end_list": answer_end_list,
            "answer_index_list": answer_index_list,
            "offset_mapping_list": offset_mapping_list,
            "batch_max_length": batch_max_length,
        }

        return batch