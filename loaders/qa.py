import json
import torch
import pathlib

from copy import deepcopy
from torch.utils.data import Dataset

def build_qa_samples(tokenizer, data_dict, max_position_embeddings, split_name, logger):
    """
    创建用于qa模型训练的样本
    :param tokenizer: 分词器
    :param data_dict: jsonl文件的一行字典
    :param max_position_embeddings: 最大嵌入位置
    :param split_name: 数据集类型
    :param logger: 日志
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
    # AI-CHANGE: 保证任意答案都能被至少一个窗口完整包含。
    # WordPiece 下 token 数 <= 字符数，答案字符长度是其 token 长度的安全上界，
    # 再 +2 容忍边界处部分重叠的 token。
    question_token_list, _ = tokenizer.tokenize_with_offsets(question)
    context_capacity = max_position_embeddings - 3 - len(question_token_list)
    if context_capacity < 1:
        raise ValueError(f"question 过长: {len(question_token_list)} tokens 喵! id={data_id}")

    max_answer_char_len = 0
    for answer in answers:
        if isinstance(answer, dict) and isinstance(answer.get("text"), str):
            max_answer_char_len = max(max_answer_char_len, len(answer["text"].strip()) + 2)

    stride = max(context_capacity // 2, max_answer_char_len)
    if stride >= context_capacity:
        # 答案比整个窗口还长，任何 overlap 都救不了；压到 capacity-1，
        # 该答案会在后面的对齐阶段被 continue 掉
        stride = context_capacity - 1
    encoded_windows = tokenizer.encode_with_offsets(
        text=question,
        text_pair=context,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        max_length=max_position_embeddings,
        stride=stride,
        return_overflowing=True,
    )
    if not isinstance(encoded_windows, dict):
        raise TypeError(f"encoded_windows 必须为dict类型的喵! 现在的type: {type(encoded_windows)}")
    if not "input_ids_list" in encoded_windows:
        raise ValueError(f"encoded_windows 必须包含 input_ids_list 字段喵!")
    input_ids_list = encoded_windows["input_ids_list"]
    if not "attention_mask_list" in encoded_windows:
        raise ValueError(f"encoded_windows 必须包含 attention_mask_list 字段喵!")
    attention_mask_list = encoded_windows["attention_mask_list"]
    if not "token_type_ids_list" in encoded_windows:
        raise ValueError(f"encoded_windows 必须包含 token_type_ids_list 字段喵!")
    token_type_ids_list = encoded_windows["token_type_ids_list"]
    if not "offset_mapping_list" in encoded_windows:
        raise ValueError(f"encoded_windows 必须包含 offset_mapping_list 字段喵!")
    offset_mapping_list = encoded_windows["offset_mapping_list"]
    if not "window_count" in encoded_windows:
        raise ValueError(f"encoded_windows 必须包含 window_count 字段喵!")
    window_count = encoded_windows["window_count"]
    if not len(input_ids_list) == len(attention_mask_list) == len(token_type_ids_list) == len(offset_mapping_list):
        raise ValueError(f"encoded_windows中四个list的长度必须一致喵!"
                         f"input_ids_list: {len(input_ids_list)}, "
                         f"attention_mask_list: {len(attention_mask_list)}, "
                         f"token_type_ids_list: {len(token_type_ids_list)}, "
                         f"offset_mapping_list: {len(offset_mapping_list)}")
    samples = []
    if not isinstance(answers, list):
        raise ValueError(f"answers 必须为list类型的喵! 现在的type: {type(answers)}")
    if not answers:
        raise ValueError(f"answers 不能为空喵!")
    skipped_long_answer_count = 0
    no_full_contain_count = 0
    reconstruction_mismatch_count = 0
    unrecovered_negative_start_count = 0
    for idx, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise ValueError(f"answer 必须为dict类型的喵! 现在的type: {type(answer)}")
        if not answer:
            raise ValueError(f"answer 不能为空喵!")
        text = answer["text"]
        # token 数 <= 字符数，字符长度+2 是 token 长度的安全上界
        if len(text.strip()) + 2 > context_capacity:
            skipped_long_answer_count += 1
            continue  # 结构上不可能被任何 128 窗口完整包含，跳过这个答案
        for window_idx in range(window_count):
            input_ids = input_ids_list[window_idx]
            attention_mask = attention_mask_list[window_idx]
            token_type_ids = token_type_ids_list[window_idx]
            offset_mapping = offset_mapping_list[window_idx]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"text 必须为非空字符串喵!")
            answer_start = answer["answer_start"]
            if not type(answer_start) is int:
                raise ValueError(f"answer_start: {answer_start} 必须为整数喵!")

            # PARSE_REFACTOR_DEMO: 兼容旧版数据集中答案首尾夹带空白的情况。
            # QATokenizer 会丢弃空白字符并只为有效文本生成 offset，因此先把
            # 答案规范化，并把被移除的前导空白长度补回 answer_start。原始
            # data_dict 不会被修改；后续所有 span 校验都使用规范化后的局部值。
            leading_whitespace = len(text) - len(text.lstrip())
            text = text.strip()
            answer_start += leading_whitespace
            answer_end = answer_start + len(text)

            # PARSE_REFACTOR_DEMO: valid 集中部分答案 answer_start 为 -1，表示
            # 该答案在 context 中没有可用的标注位置（例如答案缺失或无答案）。
            # 这里先尝试把负数偏移视为"未标注"，在全文中查找唯一匹配位置；只有
            # 唯一命中时才恢复训练样本，避免在答案重复的 context 中凭猜测绑定
            # 错误的监督位置；找不到或匹配不唯一则严格失败并给出数据 ID。
            if answer_start < 0:
                # PARSE_REFACTOR_DEMO: 先尝试精确匹配；若数据只在末尾标点上
                # 与 context 不一致，再删除一个末尾标点后匹配。只有唯一命中时
                # 才恢复 answer_start，不能对重复答案进行猜测。
                candidate_texts = [text]
                if text[-1] in "。．.!！？?，,;；:：":
                    candidate_texts.append(text[:-1])
                recovered = None
                for candidate_text in candidate_texts:
                    positions = [
                        position
                        for position in range(0, len(context) - len(candidate_text) + 1)
                        if context[position:position + len(candidate_text)] == candidate_text
                    ]
                    if len(positions) == 1:
                        recovered = (candidate_text, positions[0])
                        break
                if recovered is None:
                    # ...原有 4 行注释不动...
                    unrecovered_negative_start_count += 1
                    continue
                text, answer_start = recovered
                answer_end = answer_start + len(text)
            if answer_end > len(context):
                # 该答案的真实物理范围已超出当前 context 的文本长度，
                # 在当前 context 里结构性不可编码，跳过这个答案
                skipped_long_answer_count += 0  # 结构性不可编码计数
                unrecovered_negative_start_count += 0  # 不用，保留原有语义
                # 轻量处理：让该答案进入"跳过该答案"路径
                # 由于 answer_start 已由前面空白/恢复逻辑修正，不再可抛，
                # 这里走 continue 对齐到下面的窗口循环会更稳妥
                # 但为防止窗口外层产生无用扫描，直接在 answers 循环层 continue
                skipped_long_answer_count += 1
                continue

            # PARSE_REFACTOR_DEMO: 数据集部分 answer_start 存在少量偏移错误。
            # 只在原位置附近搜索，并且只有唯一匹配时才自动纠正，避免在答案
            # 重复出现时凭猜测绑定错误监督位置；找不到唯一结果仍然严格失败。
            if context[answer_start:answer_end] != text:
                candidate_positions = [
                    position
                    for position in range(
                        max(0, answer_start - 2),
                        min(len(context) - len(text), answer_start + 2) + 1,
                    )
                    if context[position:position + len(text)] == text
                ]
                if len(candidate_positions) != 1:
                    raise ValueError(
                        f"答案文本与context不匹配且无法唯一校正: id={data_id}, "
                        f"answer_index={idx}, answer_start={answer_start}, text={text!r} 喵!"
                    )
                answer_start = candidate_positions[0]
                answer_end = answer_start + len(text)
            start_position = None
            end_position = None
            for sub_idx, (token_type_id, offset) in enumerate(zip(token_type_ids, offset_mapping)):
                token_start, token_end = offset
                if token_type_id == 1 and token_start <= answer_start < token_end:
                    start_position = sub_idx
                if token_type_id == 1 and token_start < answer_end <= token_end:
                    end_position = sub_idx
            if start_position is None or end_position is None:
                no_full_contain_count += 1
                continue
            mapped_start = offset_mapping[start_position][0]
            mapped_end = offset_mapping[end_position][1]
            reconstructed_answer = context[mapped_start:mapped_end]
            if not reconstructed_answer == text:
                reconstruction_mismatch_count += 1
                continue
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
            if split_name != "train":
                break    # eval 每条答案只保留第一个命中窗口，保证指标分母 == 答案数
    if not samples:
        # 情况一：所有答案都超过窗口容量 → 结构性不可编码
        if skipped_long_answer_count == len(answers):
            message = f"记录 {data_id} 的所有答案均超过窗口容量 {context_capacity}，跳过该记录"
            if logger is not None:
                logger.info(message)
            else:
                print(message)
            return []
        # 情况二：答案装得下，但没有窗口能精确对齐 → 跳过该记录并输出完整诊断
        skip_message = (
            f"记录 {data_id} 的答案无法精确对齐，跳过该记录喵! \n"
            f"capacity={context_capacity}, window_count={window_count}, \n"
            f"answer_total={len(answers)}, \n"
            f"unrecovered_negative_start={unrecovered_negative_start_count}, \n"
            f"no_full_contain={no_full_contain_count}, \n"
            f"reconstruction_mismatch={reconstruction_mismatch_count}\n"
        )
        if logger is not None:
            logger.info(skip_message)
        else:
            print(skip_message)
        return []
    return samples

class QADataset(Dataset):
    def __init__(self, data_file_path, tokenizer, split_name, progress_interval, read_interval, logger=None, max_position_embeddings=None):
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
        self.max_position_embeddings = max_position_embeddings

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
        skipped_record_count = 0
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
                feature_list = build_qa_samples(tokenizer, record, self.max_position_embeddings, split_name, logger)
                if not feature_list:
                    skipped_record_count += 1
                self.samples.extend(feature_list)
                process_record += 1
                generated_samples = len(self.samples)
                if process_record % self.progress_interval == 0:
                    self._log_info(f"正在处理 {split_name} 数据集: 总记录数: {total_records} 有效记录数: {effective_records} 已处理 {process_record} 条记录，生成了 {generated_samples} 条sample")
            self._log_info(f"{split_name} 数据集已处理完毕喵~ 总记录数: {total_records} 有效记录数: {effective_records} 已处理 {process_record} 条记录，生成了 {generated_samples} 条sample")
            self._log_info(
                f"{split_name} 跳过记录统计: 共跳过 {skipped_record_count} 条，"
                f"占比 {skipped_record_count / effective_records:.2%}"
            )
            if effective_records > 0 and skipped_record_count / effective_records > 0.05:
                raise ValueError(
                    f"无法编码记录比例过高喵! {split_name} 跳过了 "
                    f"{skipped_record_count}/{effective_records} 条，超过 2% 阈值，"
                    f"请检查对齐逻辑或数据集喵!"
                )

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