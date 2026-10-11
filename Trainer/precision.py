import torch
import torch.distributed as dist

from contextlib import nullcontext
from copy import deepcopy
from math import isfinite

def validate_precision_state(state):
    if not isinstance(state, dict):
        raise TypeError("state 必须为dict类型的喵! 现在的type")
    expected_keys = {"precision", "scaler_state"}
    missing_keys = expected_keys - set(state.keys())
    extra_keys = set(state.keys()) - expected_keys
    if missing_keys:
        raise ValueError(f"当前state中缺少了键: {missing_keys} 喵!")
    if extra_keys:
        raise ValueError(f"当前state中多出了键: {extra_keys} 喵!")
    precision = state['precision']
    scaler_state = state['scaler_state']
    if not isinstance(precision, str):
        raise TypeError(f"precision 必须为str类型的喵!")
    if not precision in ["fp32", "fp16", "bf16"]:
        raise ValueError(f"当前精度precision: {precision} 不受支持喵! 受支持的有: fp32, fp16, bf16喵!")
    if precision != "fp16":
        if not scaler_state is None:
            raise ValueError(f"precision 为fp16时, scaler_state: {scaler_state} 必须为None喵!")
        else:
            return deepcopy(state)
    else:
        if not isinstance(scaler_state, dict):
            raise TypeError(f"scaler_state: {scaler_state} 必须为dict类型的喵! 现在的type: {type(scaler_state)}")
        if not scaler_state:
            raise ValueError(f"scaler_state 不能为空喵!")
        expected_scaler_keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
        missing_scaler_keys = expected_scaler_keys - set(scaler_state.keys())
        extra_scaler_keys = set(scaler_state.keys()) - expected_scaler_keys
        if missing_scaler_keys:
            raise ValueError(f"scaler_state 中缺失了键: {missing_scaler_keys} 喵!")
        if extra_scaler_keys:
            raise ValueError(f"scaler_state 中多出了键: {extra_scaler_keys} 喵!")
        scale = scaler_state['scale']
        growth_factor = scaler_state['growth_factor']
        backoff_factor = scaler_state['backoff_factor']
        growth_interval = scaler_state['growth_interval']
        _growth_tracker = scaler_state['_growth_tracker']
        if not type(scale) in (int, float) or not isfinite(scale):
            raise ValueError(f"scale: {scale} 必须为有限整型或浮点型数值喵!")
        if not type(growth_factor) in (int, float) or not isfinite(growth_factor):
            raise ValueError(f"growth_factor: {growth_factor} 必须为有限整型或浮点型数值喵!")
        if not type(backoff_factor) in (int, float) or not isfinite(backoff_factor):
            raise ValueError(f"backoff_factor: {backoff_factor} 必须为有限整型或浮点型数值喵!")
        if not scale > 0:
            raise ValueError(f"scale: {scale} 必须为大于0的数值喵!")
        if not growth_factor > 1:
            raise ValueError(f"growth_factor: {growth_factor} 必须为大于1的数值喵!")
        if not 0 < backoff_factor < 1:
            raise ValueError(f"backoff_factor: {backoff_factor} 必须为0与1之间的数值喵!")
        if not type(growth_interval) is int or not growth_interval > 0:
            raise ValueError(f"growth_interval: {growth_interval} 必须为正整数喵!")
        if not type(_growth_tracker) is int or not 0<=_growth_tracker<growth_interval:
            raise ValueError(f"_growth_tracker 必须为整型喵! 且必须大于等于0，小于growth_interval: {growth_interval} 喵!")
        return deepcopy(state)

class PrecisionController:

    def __init__(self, device, precision):
        if not isinstance(device, torch.device):
            raise TypeError(f"device: {device} 必须为 torch.device 类型的喵! 现在的type: {type(device)}")
        device_type = device.type
        if device_type not in ['cpu', 'cuda']:
            raise ValueError(f"device_type: {device_type} 必须为cpu或cuda喵!")
        if not isinstance(precision, str):
            raise TypeError(f"precision: {precision} 必须为str类型的喵! 现在的type: {type(precision)}")
        if not precision in ["fp32", "fp16", "bf16"]:
            raise ValueError(f"precision: {precision} 必须为fp32, 或fp16, 或bf16 喵!")
        if device_type == "cpu" and precision == "fp16":
            raise ValueError(f"device 为cpu时, precision不能为fp16喵!")
        self.device = device
        self.precision = precision
        if precision == "fp32":
            self.autocast_dtype = None
        elif precision == "fp16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.bfloat16
        if device_type == "cuda" and precision == "fp16":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    def autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def backward(self, loss):
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"loss 必须为torch.Tensor类型的喵! 现在的type: {type(loss)}")
        if not loss.dim() == 0:
            raise ValueError(f"loss 必须为0维的Tensor喵! 现在的维度: {loss.dim()}")
        if not torch.is_floating_point(loss):
            raise ValueError(f"loss 必须为浮点类型的Tensor喵! 现在的dtype: {loss.dtype}")
        if self.scaler is None:
            loss.backward()
        else:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        return None

    def step(self, optimizer, runtime_context):
        context = runtime_context

        def collect_description():
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            has_grad = False
            finite = True
            for group in optimizer.param_groups:
                for param in group['params']:
                    grad = param.grad
                    if grad is None:
                        continue
                    else:
                        if not grad.layout == torch.strided:
                            raise ValueError(f"当前grad不为密集梯度喵!")
                    has_grad = True
                    grad_finite = bool(torch.isfinite(grad).all().item())
                    finite = finite and grad_finite
            if not has_grad:
                raise ValueError(f"没有检测到grad喵!")
            scaler_state = None
            if self.scaler is not None:
                scaler_state = self.scaler.state_dict()
            return {"has_grad": has_grad, "finite": finite, "scaler_state": scaler_state}
        description = context.run_local("检查更新前梯度", collect_description)
        if context.enabled:
            descriptions = [None] * context.world_size
            dist.all_gather_object(descriptions, description)
            for d in descriptions:
                if not d == descriptions[0]:
                    diff_keys = []
                    for same_key in set(d.keys()) | set(descriptions[0].keys()):
                        if d.get(same_key) != descriptions[0].get(same_key):
                            diff_keys.append(same_key)
                    raise ValueError(f"DDP配置不一致喵! rank: {d} 差异字段: {diff_keys}")
        if not description["finite"] and self.scaler is None:
            raise RuntimeError(f"当前梯度不为有限值且无法进行裁剪喵!")

        def perform_update():
            did_update = False
            def on_step(optimizer_arg, args, kwargs):
                nonlocal did_update
                did_update = True
                return None
            handle = optimizer.register_step_post_hook(on_step)
            try:
                if self.scaler is None:
                    optimizer.step()
                else:
                    self.scaler.step(optimizer)
                    self.scaler.update()
            finally:
                handle.remove()
            scaler_state = None
            if self.scaler is not None:
                scaler_state = self.scaler.state_dict()
            return {
                'did_update': did_update,
                'scaler_state': scaler_state
            }
        update_description = context.run_local("执行参数更新", perform_update)
        if context.enabled:
            update_descriptions = [None] * context.world_size
            dist.all_gather_object(update_descriptions, update_description)
            for d in update_descriptions:
                if not d == update_descriptions[0]:
                    diff_keys = []
                    for same_key in set(d.keys()) | set(update_descriptions[0].keys()):
                        if d.get(same_key) != update_descriptions[0].get(same_key):
                            diff_keys.append(same_key)
                    raise ValueError(f"DDP配置不一致喵! rank: {d} 差异字段: {diff_keys}")
        did_update = update_description['did_update']
        return did_update

    def state_dict(self):
        scaler_state = None
        if self.scaler is not None:
            scaler_state = deepcopy(self.scaler.state_dict())
        state = {
            'precision': self.precision,
            'scaler_state': scaler_state
        }
        validated_state = validate_precision_state(state)
        return validated_state

