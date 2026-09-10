import torch

class StepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super(StepLR, self).__init__(optimizer=optimizer,
                                     step_size=step_size,
                                     gamma=gamma,
                                     last_epoch=last_epoch)

class WarmupScheduler(object):
    def __init__(self, optimizer, base_scheduler_type, base_scheduler_params, warmup_steps):
        super(WarmupScheduler, self).__init__()
        from models.Transformer.scheduler.registry import SCHEDULER_REGISTRY
        SCHEDULER_CANONICAL_MAP = SCHEDULER_REGISTRY.canonical_map
        if not type(warmup_steps) is int or warmup_steps < 0:
            raise ValueError(f"warmup_steps: {warmup_steps} 必须为非负int类型的喵! 现在的type: {type(warmup_steps)}")
        if not isinstance(base_scheduler_type, str) or not base_scheduler_type.strip():
            raise ValueError(f"base_scheduler_type: {base_scheduler_type} 必须为非空字符串类型的喵!")
        if not isinstance(base_scheduler_params, dict):
            raise ValueError(f"base_scheduler_params: {base_scheduler_params} 必须为dict类型的喵!")
        canonical_key = base_scheduler_type.strip().casefold()
        if canonical_key not in SCHEDULER_CANONICAL_MAP:
            raise ValueError(
                f"base_scheduler_type: {base_scheduler_type} 不受支持喵! "
                f"支持的调度器: {sorted(SCHEDULER_CANONICAL_MAP.keys())}"
            )
        scheduler_type = SCHEDULER_CANONICAL_MAP[canonical_key]
        if "last_epoch" in base_scheduler_params:
            raise ValueError(f"base_scheduler_params 不允许包含 last_epoch 喵! 它由 WarmupScheduler 内部管理")
        self.base_scheduler = scheduler_type(optimizer, **base_scheduler_params)
        self._step_count = 0
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.optimizer = optimizer
        if self.warmup_steps > 0:
            initial_factor = 1 / self.warmup_steps
        else:
            initial_factor = 1.0
        current_lr = self.base_lr * initial_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self._last_lr = current_lr
        self._base_scheduler_canonical_key = canonical_key

    def step(self):
        self._step_count += 1
        if self._step_count < self.warmup_steps:
            current_lr = self.base_lr * (self._step_count + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            self._last_lr = current_lr
        else:
            self.base_scheduler.step()
            self._last_lr = self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        return [self._last_lr]

    def state_dict(self):
        state_dict = {
            "step_count": self._step_count,
            "last_lr": self._last_lr,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "base_scheduler_state": self.base_scheduler.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            raise ValueError(f"state_dict 必须为dict类型的喵! 现在的type: {type(state_dict)}")
        for key in ("step_count", "last_lr", "base_lr", "warmup_steps", "base_scheduler_state"):
            if key not in state_dict:
                raise ValueError(f"state_dict 缺少字段: {key} 喵!")
        if state_dict["warmup_steps"] != self.warmup_steps:
            raise ValueError(
                f"warmup_steps 不匹配喵! 当前: {self.warmup_steps} 存档: {state_dict['warmup_steps']}"
            )
        self._step_count = state_dict["step_count"]
        self.base_lr = state_dict["base_lr"]
        self.base_scheduler.load_state_dict(state_dict["base_scheduler_state"])
        if self._step_count < self.warmup_steps:
            if self.warmup_steps > 0:
                warmup_factor = (self._step_count + 1) / self.warmup_steps
            else:
                warmup_factor = 1.0
            current_lr = self.base_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            self._last_lr = current_lr
        else:
            self._last_lr = self.optimizer.param_groups[0]['lr']

