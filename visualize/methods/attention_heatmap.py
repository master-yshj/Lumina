import torch
import pathlib
import math
import warnings
import matplotlib.pyplot as plt

import torch.nn as nn
import numpy as np

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
        self._handles.clear()
        self.captured.clear()

    def _resolve_layers(self, num_layers):
        if not type(num_layers) is int or num_layers <= 0:
            raise ValueError(f"num_layers: {num_layers} 必须为正整数喵!")
        if type(self.layer) is int:
            if self.layer >= 0:
                resolved_layer = self.layer
                if not 0 <= resolved_layer < num_layers:
                    raise ValueError(f"layer: {self.layer} 索引越界了喵!")
                return [resolved_layer]
            else:
                resolved_layer = num_layers + self.layer
                if not 0 <= resolved_layer < num_layers:
                    raise ValueError(f"layer: {self.layer} 索引越界了喵!")
                return [resolved_layer]
        elif isinstance(self.layer, list):
            resolved_layers = []
            for single_layer in self.layer:
                if single_layer >= 0:
                    resolved_layer = single_layer
                    if not 0 <= resolved_layer < num_layers:
                        raise ValueError(f"layer: {self.layer} 越界了喵!")
                    resolved_layers.append(resolved_layer)
                else:
                    resolved_layer = single_layer + num_layers
                    if not 0 <= resolved_layer < num_layers:
                        raise ValueError(f"layer: {self.layer} 越界了喵!")
                    resolved_layers.append(resolved_layer)
            return resolved_layers
        elif self.layer == "all":
            resolved_layers = []
            for idx in range(num_layers):
                resolved_layers.append(idx)
            return resolved_layers
        else:
            raise ValueError(f"layer: {self.layer} 的解析出现错误了喵!")

    def _resolve_heads(self, num_heads):
        if not type(num_heads) is int or num_heads <= 0:
            raise ValueError(f"num_heads: {num_heads} 必须为正整数喵!")
        if type(self.head) is int:
            if self.head >= 0:
                resolved_head = self.head
                if not 0 <= resolved_head < num_heads:
                    raise ValueError(f"head: {self.head} 越界了喵!")
                return [resolved_head]
            else:
                resolved_head = num_heads + self.head
                if not 0 <= resolved_head < num_heads:
                    raise ValueError(f"head: {self.head} 越界了喵!")
                return [resolved_head]
        elif isinstance(self.head, list):
            resolved_heads = []
            for single_head in self.head:
                if single_head >= 0:
                    resolved_head = single_head
                    if not 0 <= resolved_head < num_heads:
                        raise ValueError(f"head: {self.head} 越界了喵!")
                    resolved_heads.append(resolved_head)
                else:
                    resolved_head = single_head + num_heads
                    if not 0 <= resolved_head < num_heads:
                        raise ValueError(f"head: {self.head} 越界了喵!")
                    resolved_heads.append(resolved_head)
            return resolved_heads
        elif self.head == "all":
            resolved_heads = []
            for idx in range(num_heads):
                resolved_heads.append(idx)
            return resolved_heads
        else:
            raise ValueError(f"head: {self.head} 的解析出现错误了喵!")

    def _get_valid_tokens(self, input_ids_row, attention_mask_row, tokenizer):
        seq_valid = int(attention_mask_row.sum().item())
        seq_length = input_ids_row.size(0)
        if not 0 < seq_valid <= seq_length:
            raise ValueError(f"seq_valid: {seq_valid} 必须大于0小于等于 seq_length: {seq_length} 喵!")
        valid_input_ids = input_ids_row[:seq_valid].tolist()
        valid_tokens = tokenizer.convert_ids_to_tokens(valid_input_ids)
        if not len(valid_tokens) == seq_valid:
            raise ValueError(f"valid_tokens 的长度: {len(valid_tokens)} 必须等于 seq_valid: {seq_valid} 喵!")
        return (valid_tokens, seq_valid)

    def _plot_single_heatmap(self, matrix, tokens, title, save_path):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"matrix 必须为 numpy 二维数组喵! 现在的type: {type(matrix)}")
        if not matrix.ndim == 2:
            raise ValueError(f"matrix 必须为二维数组喵! 现在的ndim: {matrix.ndim}")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"tokens 必须为非空 list 喵! 现在是: {tokens}")
        if not all(type(t) is str for t in tokens):
            raise TypeError(f"tokens 中的每个元素必须为 str 喵! 现在是: {tokens}")
        if not matrix.shape[0] == matrix.shape[1] == len(tokens):
            raise ValueError(f"matrix 的形状: {matrix.shape} 必须与 tokens 长度: {len(tokens)} 一致喵!")
        if not type(title) is str or not title:
            raise ValueError(f"title 必须为非空字符串喵! 现在是: {title}")
        save_path = pathlib.Path(save_path)
        if save_path.parent.as_posix() not in ("", "."):
            save_path.parent.mkdir(parents=True, exist_ok=True)

        s_valid = len(tokens)
        raw_vmax = float(np.max(matrix))
        if not math.isfinite(raw_vmax) or raw_vmax <= 0:
            vmax = 1.0
        else:
            vmax = min(1.0, raw_vmax)
        fig_w = min(12.0, max(8.0, s_valid * 0.12))
        fig_h = min(10.0, max(6.0, s_valid * 0.10))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        try:
            im = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap="Blues", aspect="auto", interpolation="nearest")
            max_ticks = 30
            tick_step = max(1, math.ceil(s_valid / max_ticks))
            tick_idx = list(range(0, s_valid, tick_step))
            tick_labels = [tokens[i] for i in tick_idx]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_xlabel("Key tokens")
            ax.set_ylabel("Query tokens")
            ax.set_title(f"{title} (max={raw_vmax:.3f})")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(save_path, dpi=150)
        finally:
            plt.close(fig)

    def _setup_chinese_font(self, logger):
        candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "PingFang SC", "Microsoft YaHei"]
        try:
            from matplotlib import font_manager
            available = {f.name for f in font_manager.fontManager.ttflist}
            for name in candidates:
                if name in available:
                    plt.rcParams["font.sans-serif"] = [name]
                    plt.rcParams["axes.unicode_minus"] = False
                    return name
        except Exception:
            pass
        try:
            logger.info("未找到可用的中文字体喵! 中文 token 可能显示为方块字, 英文不受影响喵!")
        except Exception:
            pass
        return None

    def visualize(self, model, tokenizer, dataloader, save_dir, device, logger):
        if isinstance(save_dir, str):
            if not save_dir:
                raise ValueError(f"save_dir: {save_dir} 为str类型时不能为空喵!")
        save_dir = pathlib.Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(tokenizer, "convert_ids_to_tokens"):
            raise ValueError(f"tokenizer 必须具有convert_ids_to_tokens方法喵!")
        self._setup_chinese_font(logger)
        device = torch.device(device)
        was_training = model.training
        model.eval()
        self._install_hooks(model)
        num_layers = len(self._handles)
        layer_ids = self._resolve_layers(num_layers)
        try:
            with torch.no_grad():
                head_ids = None
                visualized_sample_count = 0
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch["token_type_ids"].to(device)
                    self.captured.clear()
                    model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    if not len(self.captured) == num_layers:
                        raise RuntimeError(f"captured的长度: {len(self.captured)} 必须与 num_layers: {num_layers} 一致喵!")
                    if head_ids is None:
                        num_heads = next(iter(self.captured.values())).shape[1]
                        head_ids = self._resolve_heads(num_heads)
                    batch_size = input_ids.size(0)
                    for sample_idx_in_batch in range(batch_size):
                        if visualized_sample_count >= self.max_samples:
                            break
                        valid_tokens, valid_seq_len = self._get_valid_tokens(
                            input_ids[sample_idx_in_batch].cpu(),
                            attention_mask[sample_idx_in_batch].cpu(),
                            tokenizer,
                        )
                        for resolved_layer_idx in layer_ids:
                            for resolved_head_idx in head_ids:
                                valid_attention_matrix = self.captured[resolved_layer_idx][sample_idx_in_batch, resolved_head_idx][
                                    :valid_seq_len, :valid_seq_len].numpy()
                                heatmap_save_path = save_dir / f"sample_{visualized_sample_count:03d}_layer{resolved_layer_idx}_head{resolved_head_idx}.png"
                                heatmap_title = f"sample {visualized_sample_count} layer {resolved_layer_idx} head {resolved_head_idx}"
                                self._plot_single_heatmap(valid_attention_matrix, valid_tokens, heatmap_title, heatmap_save_path)
                        visualized_sample_count += 1
                        logger.info(f"可视化已完成单个样本喵! visualized_sample_count: {visualized_sample_count}, valid_seq_len: {valid_seq_len}, layer_ids: {layer_ids}, head_ids: {head_ids}")
                    if visualized_sample_count >= self.max_samples:
                        break
        finally:
            self._remove_hooks()
            if was_training:
                model.train()