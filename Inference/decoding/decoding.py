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