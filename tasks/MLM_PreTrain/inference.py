import torch

from math import isfinite

class MLMInference:
    def __init__(self, model, tokenizer, top_k, max_seq_length):
        if not type(top_k) is int or top_k <= 0 or not isfinite(top_k):
            raise ValueError(f"top_k: {top_k} 必须为有限正整数喵!")
        if not top_k <= len(tokenizer):
            raise ValueError(f"top_k: {top_k} 必须小于等于vocab_size: {len(tokenizer)} 喵!")
        if not type(max_seq_length) is int or max_seq_length <= 0 or not isfinite(max_seq_length):
            raise ValueError(f"max_seq_length: {max_seq_length} 必须为有限正整数喵!")


        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.max_seq_length = max_seq_length


    def infer(self, text):
        if not isinstance(text, str):
            raise ValueError(f"text必须为str类型的喵! 现在的type: {type(text)}")
        if not text.strip():
            raise ValueError(f"text不能为空喵!")
        if self.tokenizer.mask_token not in text:
            raise ValueError(f"text中没有出现 [MASK] 掩码喵! 无法进行预测喵!")
        encoded_dict = self.tokenizer.encode(preserve_special_tokens=True, text=text)
        if not isinstance(encoded_dict, dict):
            raise ValueError(f"encoded_dict 必须为dict类型的喵! 现在的type: {type(encoded_dict)}")
        if not encoded_dict:
            raise ValueError(f"encoded_dict 不能为空喵!")
        if "input_ids" not in encoded_dict:
            raise ValueError(f"encoded_dict 必须含有字段input_ids喵!")
        if "token_type_ids" not in encoded_dict:
            raise ValueError(f"encoded_dict 必须含有字段token_type_ids喵!")
        if "attention_mask" not in encoded_dict:
            raise ValueError(f"encoded_dict 必须含有字段attention_mask喵!")
        input_ids = encoded_dict["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError(f"input_ids 必须为list类型的喵! 现在的type: {type(input_ids)}")
        if not input_ids:
            raise ValueError(f"input_ids 不能为空喵!")
        if len(input_ids) > self.max_seq_length:
            raise ValueError(f"input_ids 的长度: {len(input_ids)} 不能超过max_seq_length: {self.max_seq_length}喵!")
        attention_mask = encoded_dict["attention_mask"]
        if not isinstance(attention_mask, list):
            raise ValueError(f"attention_mask 必须为list类型的喵! 现在的type: {type(attention_mask)}")
        if not attention_mask:
            raise ValueError(f"attention_mask 不能为空喵!")
        if len(attention_mask) > self.max_seq_length:
            raise ValueError(f"attention_mask 的长度: {len(attention_mask)} 不能超过max_seq_length: {self.max_seq_length}喵!")
        token_type_ids = encoded_dict["token_type_ids"]
        if not isinstance(token_type_ids, list):
            raise ValueError(f"token_type_ids 必须为list类型的喵! 现在的type: {type(token_type_ids)}")
        if not token_type_ids:
            raise ValueError(f"token_type_ids 不能为空喵!")
        if len(token_type_ids) > self.max_seq_length:
            raise ValueError(f"token_type_ids 的长度: {len(token_type_ids)} 不能超过max_seq_length: {self.max_seq_length}喵!")

        if not (len(input_ids) == len(token_type_ids) == len(attention_mask)):
            raise RuntimeError(f"input_ids, token_type_ids, attention_mask, 三个值的长度不一致喵！"
                               f"input_ids长度为：{len(input_ids)}"
                               f"token_type_ids长度为：{len(token_type_ids)}"
                               f"attention_mask长度为：{len(attention_mask)}")

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        mask_positions = [
            idx
            for idx, token_id in enumerate(input_ids)
            if token_id == self.tokenizer.mask_token_id
        ]

        if not len(mask_positions) == text.count(self.tokenizer.mask_token):
            raise ValueError(f"tokenize后的[MASK]数和原始文本中的[MASK]数不一致喵!\n"
                             f"tokenize后: {len(mask_positions)}\n"
                             f"原始文本: {text.count(self.tokenizer.mask_token)}\n")

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        token_type_ids_tensor = torch.tensor([token_type_ids], dtype=torch.long)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)

        device = next(self.model.parameters()).device
        input_ids_tensor = input_ids_tensor.to(device)
        token_type_ids_tensor = token_type_ids_tensor.to(device)
        attention_mask_tensor = attention_mask_tensor.to(device)

        batch = {
            "input_ids": input_ids_tensor,
            "token_type_ids": token_type_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }

        logits = self.model(**batch)        # [1,S,V]
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"logits 必须为Tensor类型的喵! 现在的type: {type(logits)}")
        if not logits.dim() == 3:
            raise ValueError(f"logits 的维度必须为3维喵! 现在的维度: {logits.dim()}")
        batch_size = logits.size(0)
        seq_length = logits.size(1)
        num_classes = logits.size(2)
        if not batch_size == 1:
            raise ValueError(f"推理阶段的batch_size只能为1喵!")
        if not seq_length == len(input_ids):
            raise ValueError(f"seq_length: {seq_length} 必须和input_ids的长度: {len(input_ids)} 一致喵!")
        if not num_classes == len(self.tokenizer):
            raise ValueError(f"num_classes: {num_classes} 必须和词表大小: {len(self.tokenizer)} 一致喵!")
        if not torch.isfinite(logits).all().item():
            raise ValueError(f"logits中的所有数值都必须有限喵!")
        mask_logits = logits[0,mask_positions,:]
        pred_prob = torch.softmax(mask_logits, dim=-1)
        top_scores, top_token_ids = torch.topk(pred_prob, k=self.top_k)
        top_scores = top_scores.detach().cpu().tolist()
        top_token_ids = top_token_ids.detach().cpu().tolist()
        predictions = []
        for mask_position, single_top_scores, single_top_token_ids in zip(mask_positions, top_scores, top_token_ids):
            int_top_token_ids = []
            float_top_scores = []
            for single_top_score, single_top_token_id in zip(single_top_scores, single_top_token_ids):
                float_top_scores.append(float(single_top_score))
                int_top_token_ids.append(int(single_top_token_id))
            top_tokens = self.tokenizer.convert_ids_to_tokens(int_top_token_ids)
            if not len(int_top_token_ids) == len(float_top_scores) == len(top_tokens) == self.top_k:
                raise ValueError(f"int_top_token_ids的长度, float_top_scores的长度, top_tokens的长度, top_k 几个值的大小不一样喵!\n"
                                 f"int_top_token_ids: {len(int_top_token_ids)}\n"
                                 f"float_top_scores: {len(float_top_scores)}\n"
                                 f"top_tokens: {len(top_tokens)}\n"
                                 f"top_k: {self.top_k}\n")
            mask_summary_dict = {
                "position": mask_position,
                "top_tokens": top_tokens,
                "top_token_ids": int_top_token_ids,
                "top_scores": float_top_scores,
            }
            predictions.append(mask_summary_dict)
        if not len(predictions) == len(mask_positions):
            raise ValueError(f"模型预测结果数量与最初的[MASK]数量不一致喵!\n"
                             f"predictions: {len(predictions)}\n"
                             f"mask_positions: {len(mask_positions)}\n")
        predictions_summary = {
            "text": text,
            "tokens": tokens,
            "mask_positions": mask_positions,
            "predictions": predictions,
        }
        return predictions_summary




