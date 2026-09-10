import torch
import pathlib
import math
import matplotlib.pyplot as plt

import torch.nn as nn

from visualize.base import BaseVisualizer

class AttentionHeatmapVisualizer(BaseVisualizer):
    def __init__(self, layer=-1, head=0, max_samples=5):
        if type(layer) is int:
            pass
        elif layer == "all":
            pass
        elif isinstance(layer, list):
            if not layer:
                raise ValueError(f"layer 不能为空喵!")
            if not all(isinstance(single_layer, int) for single_layer in layer):
                raise TypeError(f"layer: {layer} 中的每个元素必须为int类型的喵!")
        else:
            raise TypeError(f"layer: {layer} 必须为 int 或 list 类型或 'all' 喵! 现在的type: {type(layer)}")
        if type(head) is int:
            pass
        elif head == "all":
            pass
        elif isinstance(head, list):
            if not head:
                raise ValueError(f"head 不能为空喵!")
            if not all(isinstance(single_head, int) for single_head in head):
                raise TypeError(f"head: {head} 中的每个元素必须为int类型的喵!")
        else:
            raise TypeError(f"head: {head} 必须为 int 或 list 类型 或 'all' 喵! 现在的type: {type(head)}")
        if not type(max_samples) is int:
            raise TypeError(f"max_samples 必须为int类型的喵! 现在的type: {type(max_samples)}")
        if not max_samples > 0:
            raise ValueError(f"max_samples: {max_samples} 必须为正整数喵!")

        super().__init__(layer=layer, head=head, max_samples=max_samples)

        self.layer = layer
        self.head = head
        self.max_samples = max_samples
        self.captured = {}
        self._handles = []

    def _find_attention_dropout_modules(self, model):
        if not isinstance(model, nn.Module):
            raise ValueError(f"model 必须为 nn.Module 类型的喵! 现在的type: {type(model)}")
        match = []
        for name, module in model.named_modules():
            if name != "" and name.split(".")[-1] == "attention_dropout":
                match.append(module)
            else:
                pass
        if not match:
            raise ValueError(f"model 中没有找到任何attention_dropout喵!")
        return match
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self.captured[layer_idx] = output.detach().cpu()
        return hook

    def _install_hooks(self, model):
        matched_modules = self._find_attention_dropout_modules(model)
        for layer_idx, module in enumerate(matched_modules):
            hook = self._make_hook(layer_idx)
            handle = module.register_forward_hook(hook)
            self._handles.append(handle)

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.captured = {}

    def visualize(self, model, tokenizer, dataloader, save_dir, device, logger):
        pass