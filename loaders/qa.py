import json
import pathlib

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