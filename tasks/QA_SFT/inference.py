import torch

class QAInference:
    def __init__(self, model, tokenizer, decoding_strategy, max_position_embeddings, doc_stride):
        if not decoding_strategy:
            raise ValueError(f"decoding_strategy 不能为空喵!")
        if not (hasattr(decoding_strategy, "decode") and callable(decoding_strategy.decode)):
            raise ValueError(f"decoding_strategy 必须有可调用的decode方法喵!")
        if not type(max_position_embeddings) is int or max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings: {max_position_embeddings} 必须为严格正整数喵!")
        if not type(doc_stride) is int or doc_stride < 0:
            raise ValueError(f"doc_stride: {doc_stride} 必须为非负整数喵!")

        self.model = model
        self.tokenizer = tokenizer
        self.decoding_strategy = decoding_strategy
        self.max_position_embeddings = max_position_embeddings
        self.doc_stride = doc_stride

    def infer(self, input_data):
        if not isinstance(input_data, dict):
            raise TypeError(f"input_data: {input_data} 必须为dict类型的喵! 现在的type: {type(input_data)}")
        if not "question" in input_data:
            raise ValueError(f"input_data: {input_data} 必须包含question字段喵!")
        if not "context" in input_data:
            raise ValueError(f"input_data: {input_data} 必须包含context字段喵!")
        question = input_data["question"]
        context = input_data["context"]
        if not isinstance(question, str):
            raise TypeError(f"question: {question} 必须为str类型的喵! 现在的type: {type(question)}")
        if not isinstance(context, str):
            raise TypeError(f"context: {context} 必须为str类型的喵! 现在的type: {type(context)}")
        if not question.strip():
            raise ValueError(f"question 不能为空喵!")
        if not context.strip():
            raise ValueError(f"context 不能为空喵!")
        encoded_windows = self.tokenizer.encode_with_offsets(
            text=question,
            text_pair=context,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            preserve_special_tokens=False,
            max_length=self.max_position_embeddings,
            stride=self.doc_stride,
            return_overflowing=True,
        )
        if not isinstance(encoded_windows, dict):
            raise TypeError(f"encoded_windows: {encoded_windows} 必须为dict类型的喵! 现在的type: {type(encoded_windows)}")
        if not encoded_windows:
            raise ValueError(f"encoded_windows 不能为空喵!")
        if not "input_ids_list" in encoded_windows:
            raise ValueError(f"encoded_windows 必须包含input_ids_list字段喵!")
        if not "attention_mask_list" in encoded_windows:
            raise ValueError(f"encoded_windows 必须包含attention_mask_list字段喵!")
        if not "token_type_ids_list" in encoded_windows:
            raise ValueError(f"encoded_windows 必须包含token_type_ids_list字段喵!")
        if not "offset_mapping_list" in encoded_windows:
            raise ValueError(f"encoded_windows 必须包含offset_mapping_list字段喵!")
        if not "window_count" in encoded_windows:
            raise ValueError(f"encoded_windows 必须包含window_count字段喵!")
        input_ids_list = encoded_windows["input_ids_list"]
        attention_mask_list = encoded_windows["attention_mask_list"]
        token_type_ids_list = encoded_windows["token_type_ids_list"]
        offset_mapping_list = encoded_windows["offset_mapping_list"]
        window_count = encoded_windows["window_count"]
        if not isinstance(input_ids_list, list):
            raise TypeError(f"input_ids_list: {input_ids_list} 必须为list类型的喵! 现在的type: {type(input_ids_list)}")
        if not isinstance(attention_mask_list, list):
            raise TypeError(f"attention_mask_list: {attention_mask_list} 必须为list类型的喵! 现在的type: {type(attention_mask_list)}")
        if not isinstance(token_type_ids_list, list):
            raise TypeError(f"token_type_ids_list: {token_type_ids_list} 必须为list类型的喵! 现在的type: {type(token_type_ids_list)}")
        if not isinstance(offset_mapping_list, list):
            raise TypeError(f"offset_mapping_list: {offset_mapping_list} 必须为list类型的喵! 现在的type: {type(offset_mapping_list)}")
        if not type(window_count) is int or window_count <= 0:
            raise ValueError(f"window_count: {window_count} 必须为严格正整数喵!")
        if not len(input_ids_list) == len(attention_mask_list) == len(token_type_ids_list) == len(offset_mapping_list) == window_count:
            raise ValueError(f"input_ids_list的长度: {len(input_ids_list)}\n"
                             f"attention_mask_list的长度: {len(attention_mask_list)}\n"
                             f"token_type_ids_list的长度: {len(token_type_ids_list)}\n"
                             f"offset_mapping_list的长度: {len(offset_mapping_list)}\n"
                             f"window_count: {window_count}\n"
                             f"这五个值必须一样喵!")
        candidates = []
        for count in range(window_count):
            input_ids = input_ids_list[count]
            attention_mask = attention_mask_list[count]
            token_type_ids = token_type_ids_list[count]
            offset_mapping = offset_mapping_list[count]
            if not len(input_ids) == len(attention_mask) == len(token_type_ids) == len(offset_mapping):
                raise ValueError(f"input_ids的长度: {len(input_ids)}\n"
                                 f"attention_mask的长度: {len(attention_mask)}\n"
                                 f"token_type_ids的长度: {len(token_type_ids)}\n"
                                 f"offset_mapping的长度: {len(offset_mapping)}\n"
                                 f"这四个值必须一致喵!")
            if not len(input_ids) <= self.max_position_embeddings:
                raise ValueError(f"input_ids的长度: {len(input_ids)} 不能大于 max_position_embeddings: {self.max_position_embeddings} 喵!")
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
            attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
            token_type_ids_tensor = torch.tensor([token_type_ids], dtype=torch.long)

            device = next(self.model.parameters()).device
            input_ids_tensor = input_ids_tensor.to(device)
            token_type_ids_tensor = token_type_ids_tensor.to(device)
            attention_mask_tensor = attention_mask_tensor.to(device)

            batch = {
                "input_ids": input_ids_tensor,
                "token_type_ids": token_type_ids_tensor,
                "attention_mask": attention_mask_tensor,
            }

            logits_dict = self.model(**batch)

            if not isinstance(logits_dict, dict):
                raise TypeError(f"logits_dict: {logits_dict} 必须为dict类型的喵! 现在的type: {type(logits_dict)}")
            if not logits_dict:
                raise ValueError(f"logits_dict 不能为空喵!")
            if not "start_logits" in logits_dict:
                raise ValueError(f"logits_dict: {logits_dict} 必须包含start_logits字段喵!")
            if not "end_logits" in logits_dict:
                raise ValueError(f"logits_dict: {logits_dict} 必须包含end_logits字段喵!")
            start_logits = logits_dict["start_logits"]
            end_logits = logits_dict["end_logits"]
            if not isinstance(start_logits, torch.Tensor):
                raise TypeError(f"start_logits: {start_logits} 必须为torch.Tensor类型的喵! 现在的type: {type(start_logits)}")
            if not isinstance(end_logits, torch.Tensor):
                raise TypeError(f"end_logits: {end_logits} 必须为torch.Tensor类型的喵! 现在的type: {type(end_logits)}")
            if not start_logits.shape == end_logits.shape:
                raise ValueError(f"start_logits的形状: {start_logits.shape}\n"
                                 f"end_logits的形状: {end_logits.shape}\n"
                                 f"两者必须一致喵!")
            batch_size = start_logits.shape[0]
            seq_length = start_logits.shape[1]
            if not batch_size == 1:
                raise ValueError(f"batch_size: {batch_size} 必须为1喵!")
            if not seq_length == len(input_ids):
                raise ValueError(f"seq_length: {seq_length} 与窗口长度: {len(input_ids)} 必须一致喵!")
            if not torch.isfinite(start_logits).all():
                raise ValueError(f"start_logits: {start_logits} 不合法喵!")
            if not torch.isfinite(end_logits).all():
                raise ValueError(f"end_logits: {end_logits} 不合法喵!")
            start_logits = start_logits.reshape(-1, seq_length)
            end_logits = end_logits.reshape(-1, seq_length)
            window_candidate = self.decoding_strategy.decode(
                start_logits=start_logits[0],
                end_logits=end_logits[0],
                token_type_ids=token_type_ids_tensor[0],
                attention_mask=attention_mask_tensor[0],
                offset_mappings=offset_mapping,
                context=context,
            )
            if not isinstance(window_candidate, dict):
                raise TypeError(f"window_candidate: {window_candidate} 必须为dict类型的喵! 现在的type: {type(window_candidate)}")
            window_candidate.update(
                {
                    "window_index": count,
                }
            )
            candidates.append(window_candidate)
        if not candidates:
            raise ValueError(f"所有滑动窗口都没有生成合法的QA答案喵!")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise TypeError(f"candidates[{candidate_index}]必须为dict类型的喵! 现在的type: {type(candidate)}")
            if "score" not in candidate:
                raise KeyError(f"candidates[{candidate_index}]中缺少score字段喵!")
            if "answer" not in candidate:
                raise KeyError(f"candidates[{candidate_index}]中缺少answer字段喵!")
            if not isinstance(candidate["score"], (int, float)):
                raise TypeError(f"candidates[{candidate_index}]['score']必须为数值类型的喵!")
            if not torch.isfinite(torch.tensor(candidate["score"], dtype=torch.float64)).item():
                raise ValueError(f"candidates[{candidate_index}]['score']必须为有限值喵!")
        best_candidate = max(
            candidates,
            key=lambda candidate: candidate["score"],
        )
        result = dict(best_candidate)
        result["question"] = question
        result["context"] = context
        result["result_text"] = result["answer"]
        result["window_count"] = window_count
        return result
