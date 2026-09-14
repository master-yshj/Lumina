import torch

from math import isfinite

class TopKSamplingStrategy:
    def __init__(self, top_k = 5):
        if not type(top_k) is int or not top_k > 0 or not isfinite(top_k) :
            raise ValueError(f"top_k: {top_k} 必须为有限正整数喵!")
        self.top_k = top_k

    def decode(self, logits):
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"logits 必须为Tensor类型的喵! 现在的type: {type(logits)}")
        if not logits.dim() == 1:
            raise ValueError(f"logits 必须为一维Tensor喵! 现在的维度: {logits.dim()}")
        if not logits.numel() > 0:
            raise ValueError(f"logits 中的元素数量不能小于等于0喵!")
        if not self.top_k <= logits.numel() :
            raise ValueError(f"logits 中的元素数量必须大于等于 top_k 喵!"
                             f"logits 元素数量: {logits.numel()}"
                             f"top_k: {self.top_k}")
        if not torch.isfinite(logits).all().item():
            raise ValueError(f"logits中的所有数值都必须有限喵!")
        top_logits, top_token_ids = torch.topk(logits, k=self.top_k)
        sampling_scores = torch.softmax(top_logits, dim=-1)
        sampled_candidate_index = torch.multinomial(sampling_scores, num_samples=1)
        sampled_token_id = top_token_ids[sampled_candidate_index]

        top_logits = top_logits.detach().cpu().tolist()
        top_token_ids = top_token_ids.detach().cpu().tolist()
        sampling_scores = sampling_scores.detach().cpu().tolist()
        sampled_candidate_index = sampled_candidate_index.item()
        sampled_token_id = sampled_token_id.item()

        summary_dict = {
            "top_logits": top_logits,
            "top_token_ids": top_token_ids,
            "sampling_scores": sampling_scores,
            "selected_candidate_index": sampled_candidate_index,
            "selected_token_id": sampled_token_id,
        }

        return summary_dict

class QASpanTopKDecodingStrategy:
    def __init__(self, max_answer_length=30, n_best_size=20):
        if not type(max_answer_length) is int:
            raise TypeError(f"max_answer_length: {max_answer_length} 必须为int类型的喵! 现在的type: {type(max_answer_length)}")
        if not max_answer_length > 0:
            raise ValueError(f"max_answer_length: {max_answer_length} 必须为严格正整数喵!")
        if not type(n_best_size) is int:
            raise TypeError(f"n_best_size: {n_best_size} 必须为int类型的喵! 现在的type: {type(n_best_size)}")
        if not n_best_size > 0:
            raise ValueError(f"n_best_size: {n_best_size} 必须为严格正整数喵!")
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size

    def decode(self, start_logits, end_logits, token_type_ids, attention_mask, offset_mappings, context):
        if not isinstance(start_logits, torch.Tensor):
            raise TypeError(f"start_logits: {start_logits} 必须为 torch.Tensor 类型的喵! 现在的type: {type(start_logits)}")
        if not isinstance(end_logits, torch.Tensor):
            raise TypeError(f"end_logits: {end_logits} 必须为 torch.Tensor 类型的喵! 现在的type: {type(end_logits)}")
        if not start_logits.dim() == 1:
            raise ValueError(f"start_logits的维度: {start_logits.dim()} 必须为1喵!")
        if not end_logits.dim() == 1:
            raise ValueError(f"end_logits的维度: {end_logits.dim()} 必须为1喵!")
        if not start_logits.shape == end_logits.shape:
            raise ValueError(f"start_logits的形状: {start_logits.shape}\n"
                             f"end_logits的形状: {end_logits.shape}\n"
                             f"两者的形状必须一致喵!")
        if not torch.is_floating_point(start_logits):
            raise TypeError(f"start_logits: {start_logits} 必须为浮点类型torch.Tensor张量喵! 现在的dtype: {start_logits.dtype}")
        if not torch.is_floating_point(end_logits):
            raise TypeError(f"end_logits: {end_logits} 必须为浮点类型torch.Tensor张量喵! 现在的dtype: {end_logits.dtype}")
        if not torch.isfinite(start_logits).all():
            raise ValueError(f"start_logits: {start_logits} 的值不合法喵!")
        if not torch.isfinite(end_logits).all():
            raise ValueError(f"end_logits: {end_logits} 的值不合法喵!")
        if not start_logits.device == end_logits.device:
            raise ValueError(f"start_logits位于设备: {start_logits.device}\n"
                             f"end_logits位于设备: {end_logits.device}\n"
                             f"两者必须一致喵!")
        if not isinstance(token_type_ids, torch.Tensor):
            raise TypeError(f"token_type_ids: {token_type_ids} 必须为torch.Tensor张量喵! 现在的type: {type(token_type_ids)}")
        if not token_type_ids.dim() == 1:
            raise ValueError(f"token_type_ids的维度: {token_type_ids.dim()} 必须为1喵!")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(f"attention_mask: {attention_mask} 必须为torch.Tensor张量喵! 现在的type: {type(attention_mask)}")
        if not attention_mask.dim() == 1:
            raise ValueError(f"attention_mask的维度: {attention_mask.dim()} 必须为1喵!")
        if not token_type_ids.device == attention_mask.device:
            raise ValueError(f"token_type_ids位于设备: {token_type_ids.device}\n"
                             f"attention_mask位于设备: {attention_mask.device}\n"
                             f"两者必须一致喵!")
        if not token_type_ids.dtype == torch.long:
            raise TypeError(f"token_type_ids: {token_type_ids} 必须为整型的torch.Tensor张量喵! 现在的dtype: {token_type_ids.dtype}")
        if not attention_mask.dtype == torch.long:
            raise TypeError(f"attention_mask: {attention_mask} 必须为整型的torch.Tensor张量喵! 现在的dtype: {attention_mask.dtype}")
        if not isinstance(offset_mappings, list):
            raise TypeError(f"offset_mappings: {offset_mappings} 必须为list类型喵! 现在的type: {type(offset_mappings)}")
        if not offset_mappings:
            raise ValueError(f"offset_mappings 不能为空喵!")
        if not len(start_logits) == len(end_logits) == len(token_type_ids) == len(attention_mask) == len(offset_mappings):
            raise ValueError(f"start_logits的长度: {len(start_logits)}\n"
                             f"end_logits的长度: {len(end_logits)}\n"
                             f"token_type_ids的长度: {len(token_type_ids)}\n"
                             f"attention_mask的长度: {len(attention_mask)}\n"
                             f"offset_mappings的长度: {len(offset_mappings)}\n"
                             f"五个值的长度必须一样喵!")
        if not isinstance(context, str):
            raise TypeError(f"context: {context} 必须为str类型的喵! 现在的type: {type(context)}")
        for offset_mapping in offset_mappings:
            if not isinstance(offset_mapping, (tuple, list)):
                raise TypeError(f"offset_mapping 必须为tuple或者list类型的喵! 现在的type: {type(offset_mapping)}")
            if not len(offset_mapping) == 2:
                raise ValueError(f"offset_mapping 的长度必须为2喵! 现在的长度: {len(offset_mapping)}")
            start = offset_mapping[0]
            end = offset_mapping[1]
            if not isinstance(start, int):
                raise TypeError(f"start: {start} 必须为int类型的喵! 现在的type: {type(start)}")
            if not isinstance(end, int):
                raise TypeError(f"end的第二个数值: {end} 必须为int类型的喵! 现在的type: {type(end)}")
            if not 0 <= start <= end <= len(context):
                raise ValueError(f"start: {start} end: {end} 两个数值不合法喵! 必须满足: 0<=start<=end<=len(context)")
        valid_context_indices = []

        for index, offset_mapping in enumerate(offset_mappings):
            token_type_id = token_type_ids[index].item()
            attention_value = attention_mask[index].item()
            offset_start, offset_end = offset_mapping

            if attention_value != 1:
                continue
            if token_type_id != 1:
                continue
            if offset_start >= offset_end:
                continue
            if not 0 <= offset_start < offset_end <= len(context):
                continue

            valid_context_indices.append(index)

        if not valid_context_indices:
            raise ValueError("当前输入中没有找到合法的context token，无法解码QA答案喵!")

        candidate_count = min(
            self.n_best_size,
            len(valid_context_indices),
        )

        valid_indices_tensor = torch.tensor(
            valid_context_indices,
            dtype=torch.long,
            device=start_logits.device,
        )

        valid_start_logits = start_logits[valid_indices_tensor]
        valid_end_logits = end_logits[valid_indices_tensor]

        _, start_candidate_positions = torch.topk(
            valid_start_logits,
            k=candidate_count,
        )
        _, end_candidate_positions = torch.topk(
            valid_end_logits,
            k=candidate_count,
        )

        start_candidate_indices = [
            valid_context_indices[position.item()]
            for position in start_candidate_positions
        ]
        end_candidate_indices = [
            valid_context_indices[position.item()]
            for position in end_candidate_positions
        ]

        best_candidate = None
        best_score = 0.0
        for start_index in start_candidate_indices:
            for end_index in end_candidate_indices:
                if start_index > end_index:
                    continue
                span_length = end_index - start_index + 1
                if span_length > self.max_answer_length:
                    continue
                if token_type_ids[start_index].item() != 1:
                    continue
                if token_type_ids[end_index].item() != 1:
                    continue
                if attention_mask[start_index].item() != 1:
                    continue
                if attention_mask[end_index].item() != 1:
                    continue
                start_logit = start_logits[start_index].item()
                end_logit = end_logits[end_index].item()
                span_score = start_logit + end_logit
                start_char = offset_mappings[start_index][0]
                end_char = offset_mappings[end_index][1]
                if not 0 <= start_char < end_char <= len(context):
                    continue
                answer = context[start_char:end_char]
                if not answer.strip():
                    continue
                candidate = {
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "answer": answer,
                    "score": span_score,
                    "start_logit": start_logit,
                    "end_logit": end_logit,
                }
                if best_candidate is None:
                    best_candidate = candidate
                elif candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate
        if best_candidate is None:
            raise ValueError(
                "没有找到合法的QA答案span喵!"
            )
        return best_candidate