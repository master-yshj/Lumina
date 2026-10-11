import dataclasses
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from device import check_device

@dataclasses.dataclass
class RuntimeContext:
    device: torch.device
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    owns_process_group: bool = False

    def __post_init__(self):
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device 必须为 torch.device 类型的喵! 现在的type: {type(self.device)}")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"device.type 必须为 cpu 或 cuda，当前为 {self.device.type!r}")
        if not type(self.enabled) is bool:
            raise TypeError(f"enabled 必须为 bool 类型的喵! 现在的type: {type(self.enabled)}")
        if not type(self.rank) is int:
            raise TypeError(f"rank 必须为 int 类型的喵! 现在的type: {type(self.rank)}")
        if not type(self.local_rank) is int:
            raise TypeError(f"local_rank 必须为 int 类型的喵! 现在的type: {type(self.local_rank)}")
        if not type(self.world_size) is int:
            raise TypeError(f"world_size 必须为 int 类型的喵! 现在的type: {type(self.world_size)}")
        if not isinstance(self.backend, (str, type(None))):
            raise TypeError(f"backend 必须为 str 类型或 None 喵! 现在的type: {type(self.backend)}")
        if not type(self.owns_process_group) is bool:
            raise TypeError(f"owns_process_group 必须为 bool 类型的喵! 现在的type: {type(self.owns_process_group)}")
        if self.rank < 0:
            raise ValueError(f"rank: {self.rank} 不能为负数喵!")
        if self.local_rank < 0:
            raise ValueError(f"local_rank: {self.local_rank} 不能为负数喵!")
        if not self.world_size >= 1:
            raise ValueError(f"world_size: {self.world_size} 必须至少为1喵!")

        if not self.enabled:
            if not self.rank == self.local_rank == 0:
                raise ValueError(f"enabled 为 False 时, rank: {self.rank} 与 local_rank: {self.local_rank} 必须相等且都为0喵!")
            if self.world_size != 1:
                raise ValueError(f"enabled 为 False 时, world_size: {self.world_size} 必须为1喵!")
            if self.backend is not None:
                raise ValueError(f"enabled 为 False 时, backend: {self.backend} 必须为 None 喵!")
            if self.owns_process_group:
                raise ValueError(f"enabled 为 False 时, owns_process_group: {self.owns_process_group} 必须为False喵!")
        else:
            if not self.world_size >= 2:
                raise ValueError(f"enabled 为 True 时, world_size: {self.world_size} 必须至少为2喵!")
            if not 0 <= self.rank < self.world_size:
                raise ValueError(f"enabled 为 True 时, 必须满足: 0 <= rank < world_size 喵! rank: {self.rank}, world_size: {self.world_size}")
            if not 0 <= self.local_rank < self.world_size:
                raise ValueError(f"enabled 为 True 时, 必须满足: 0 <= local_rank < world_size 喵! local_rank: {self.local_rank}, world_size: {self.world_size}")
            if not isinstance(self.backend, str):
                raise TypeError("enabled 为 True 时, backend 必须为非空字符串喵!")
            if not self.backend.strip():
                raise ValueError(f"enabled 为 True 时, backend 必须为非空字符串喵!")
            if not (self.device.type == "cuda" and self.backend == "nccl") and not (self.device.type == "cpu" and self.backend == "gloo"):
                raise ValueError(f"device_type 为cuda时, backend 必须为nccl喵! device_type 为cpu时, backend 必须为gloo喵!\n"
                                 f"device_type: {self.device.type}\n"
                                 f"backend: {self.backend}\n")

    @property
    def is_main(self):
        return self.rank == 0

    def run_local(self, action, function):
        if not isinstance(action, str):
            raise TypeError(f"action: {action} 必须为str类型的喵! 现在的type: {type(action)}")
        if not action.strip():
            raise ValueError(f"action 不能为空喵!")
        if not callable(function):
            raise TypeError(f"function: {function} 必须为可调用对象喵!")
        if not self.enabled:
            return function()

        result = None
        local_error = None
        try:
            result = function()
        except Exception as error:
            local_error = error

        status = {
            "rank": self.rank,
            "ok": local_error is None,
            "error_type": None,
            "message": "",
        }
        if local_error is not None:
            status["error_type"] = type(local_error).__name__
            status["message"] = str(local_error)

        statuses = [None] * self.world_size
        dist.all_gather_object(statuses, status)

        for gathered_status in statuses:
            if not gathered_status["ok"]:
                message = (
                    f"action: {action}; rank: {gathered_status['rank']}; "
                    f"error_type: {gathered_status['error_type']}; "
                    f"message: {gathered_status['message']}"
                )
                if local_error is not None:
                    raise RuntimeError(message) from local_error
                raise RuntimeError(message)

        return result

    def run_main(self, action, function):
        if not isinstance(action, str):
            raise TypeError(f"action: {action} 必须为str类型的喵! 现在的type: {type(action)}")
        if not action.strip():
            raise ValueError(f"action 不能为空喵!")
        if not callable(function):
            raise TypeError(f"function: {function} 必须为可调用对象喵!")
        if not self.enabled:
            return function()

        local_error = None
        if self.is_main:
            try:
                result = function()
                envelope = {"ok": True, "result": result, "error": None}
            except Exception as error:
                local_error = error
                envelope = {
                    "ok": False,
                    "result": None,
                    "error": {
                        "rank": self.rank,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                }
            payload = [envelope]
        else:
            payload = [None]

        dist.broadcast_object_list(payload, src=0)
        envelope = payload[0]
        if envelope["ok"]:
            return envelope["result"]

        error_info = envelope["error"]
        message = (
            f"action: {action}; rank: {error_info['rank']}; "
            f"error_type: {error_info['error_type']}; "
            f"message: {error_info['message']}"
        )
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def close(self):
        if not self.owns_process_group:
            return None
        if dist.is_initialized():
            dist.destroy_process_group()
        self.owns_process_group = False
        return None
def initialize_training_runtime(config):
    run_mode = config["run"]["mode"]
    if not isinstance(run_mode, str):
        raise TypeError(f"run.mode 必须为 str，当前类型为 {type(run_mode)}")
    if run_mode != "train":
        raise ValueError(f"当前运行模式必须为 train，收到 {run_mode!r}")

    runtime_config = config["runtime"]
    if not isinstance(runtime_config, dict):
        raise TypeError(f"runtime 必须为 dict，当前类型为 {type(runtime_config)}")
    precision = runtime_config["precision"]
    if not isinstance(precision, str):
        raise TypeError(f"runtime.precision 必须为 str，当前类型为 {type(precision)}")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"runtime.precision 不受支持：{precision!r}")

    device_config = runtime_config["device"]
    if not isinstance(device_config, dict):
        raise TypeError(f"runtime.device 必须为 dict，当前类型为 {type(device_config)}")
    device_type = device_config["type"]
    if not isinstance(device_type, str):
        raise TypeError(f"runtime.device.type 必须为 str，当前类型为 {type(device_type)}")
    if device_type not in ("cpu", "cuda"):
        raise ValueError(f"runtime.device.type 必须为 cpu 或 cuda，收到 {device_type!r}")
    device_index = device_config["index"]
    if device_index is not None and (type(device_index) is not int or device_index < 0):
        raise TypeError(
            f"runtime.device.index 必须为 None 或非负的严格 int，收到 {device_index!r}"
        )

    distributed_config = runtime_config["distributed"]
    if not isinstance(distributed_config, dict):
        raise TypeError(f"runtime.distributed 必须为 dict，当前类型为 {type(distributed_config)}")
    strategy = distributed_config["strategy"]
    if not isinstance(strategy, str):
        raise TypeError(f"runtime.distributed.strategy 必须为 str，当前类型为 {type(strategy)}")
    if strategy not in ("none", "ddp"):
        raise ValueError(f"runtime.distributed.strategy 必须为 none 或 ddp，收到 {strategy!r}")
    backend = distributed_config["backend"]
    if backend is not None and not isinstance(backend, str):
        raise TypeError(f"runtime.distributed.backend 必须为 None 或 str，收到 {type(backend)}")
    if dist.is_initialized():
        raise RuntimeError("当前进程已经存在通信组，initialize_training_runtime 不接管外部通信组")

    env_keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    missing_keys = [key for key in env_keys if key not in os.environ]
    if missing_keys and len(missing_keys) != len(env_keys):
        present_keys = [key for key in env_keys if key in os.environ]
        raise ValueError(
            f"torchrun 环境变量不能部分存在；已存在：{present_keys}，缺失：{missing_keys}"
        )
    if missing_keys and strategy == "ddp":
        raise ValueError("strategy=ddp 时必须由 torchrun 提供四个进程环境变量")

    if missing_keys:
        rank = 0
        local_rank = 0
        world_size = 1
        local_world_size = 1
    else:
        env_values = {}
        for key in env_keys:
            raw_value = os.environ[key]
            try:
                env_values[key] = int(raw_value, 10)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"环境变量 {key} 必须是十进制整数，收到 {raw_value!r}"
                ) from error
        rank = env_values["RANK"]
        local_rank = env_values["LOCAL_RANK"]
        world_size = env_values["WORLD_SIZE"]
        local_world_size = env_values["LOCAL_WORLD_SIZE"]

    if rank < 0 or local_rank < 0:
        raise ValueError(f"rank/local_rank 必须非负，收到 rank={rank}, local_rank={local_rank}")
    if world_size <= 0 or local_world_size <= 0:
        raise ValueError(
            f"world_size/local_world_size 必须为正数，收到 world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )
    if rank >= world_size or local_rank >= local_world_size:
        raise ValueError(
            f"进程编号越界：rank={rank}, world_size={world_size}, "
            f"local_rank={local_rank}, local_world_size={local_world_size}"
        )

    if strategy == "none":
        if (rank, local_rank, world_size, local_world_size) != (0, 0, 1, 1):
            raise ValueError(
                "strategy=none 必须是单进程环境：rank=0、local_rank=0、"
                "world_size=1、local_world_size=1"
            )
        if backend is not None:
            raise ValueError("strategy=none 时 runtime.distributed.backend 必须为 None")
        if device_type == "cpu" and device_index is not None:
            raise ValueError("CPU 的 runtime.device.index 必须为 None")
        if device_type == "cpu":
            device_string = "cpu"
        elif device_index is None:
            device_string = "cuda"
        else:
            device_string = f"cuda:{device_index}"
    else:
        if world_size < 2 or local_world_size != world_size or rank != local_rank:
            raise ValueError(
                "当前仅支持单机一进程一卡 DDP：要求 world_size=local_world_size>=2 且 rank=local_rank"
            )
        if device_type != "cuda":
            raise ValueError("正式 DDP 只支持 CUDA 设备")
        if device_index is not None:
            raise ValueError("DDP 下 runtime.device.index 必须为 None，由 LOCAL_RANK 选择设备")
        if backend != "nccl":
            raise ValueError("正式 DDP 的 runtime.distributed.backend 必须为 nccl")
        rendezvous_keys = ("MASTER_ADDR", "MASTER_PORT")
        missing_rendezvous = [key for key in rendezvous_keys if key not in os.environ]
        if missing_rendezvous:
            raise ValueError(f"DDP 缺少通信环境变量：{missing_rendezvous}")
        master_addr = os.environ["MASTER_ADDR"].strip()
        if not master_addr:
            raise ValueError("MASTER_ADDR 不能为空")
        try:
            master_port = int(os.environ["MASTER_PORT"], 10)
        except (TypeError, ValueError) as error:
            raise ValueError("MASTER_PORT 必须是十进制整数") from error
        if not 1 <= master_port <= 65535:
            raise ValueError(f"MASTER_PORT 必须位于 1—65535，收到 {master_port}")
        device_string = f"cuda:{local_rank}"

    device_summary = check_device(device_string, gpu_nums=1)
    if not isinstance(device_summary, dict):
        raise TypeError(f"check_device 必须返回 dict，收到 {type(device_summary)}")
    if device_summary["backend"] != device_type:
        raise ValueError(
            f"设备摘要 backend 与配置不一致：summary={device_summary['backend']!r}, "
            f"config={device_type!r}"
        )
    summary_index = device_summary["device_index"]
    if device_type == "cpu":
        if summary_index is not None:
            raise ValueError(f"CPU 设备摘要的 device_index 必须为 None，收到 {summary_index!r}")
    else:
        if type(summary_index) is not int or summary_index < 0:
            raise ValueError(f"CUDA 设备摘要的 device_index 必须为非负 int，收到 {summary_index!r}")
        if device_index is not None and summary_index != device_index:
            raise ValueError(
                f"实际 CUDA 编号与配置不一致：summary={summary_index}, config={device_index}"
            )
        if strategy == "ddp":
            if summary_index != local_rank:
                raise ValueError(
                    f"实际 CUDA 编号必须等于 LOCAL_RANK：summary={summary_index}, local_rank={local_rank}"
                )
            cuda_count = device_summary["cuda_count"]
            if type(cuda_count) is not int or world_size > cuda_count:
                raise ValueError(
                    f"DDP 进程数不能超过可用 CUDA 数：world_size={world_size}, cuda_count={cuda_count}"
                )

    context_device = torch.device(device_type, summary_index)
    if device_type == "cuda":
        torch.cuda.set_device(context_device)

    device_summary = dict(device_summary)
    device_summary["rank"] = rank
    device_summary["local_rank"] = local_rank
    device_summary["world_size"] = world_size

    if precision == "fp16" and device_type == "cpu":
        raise ValueError("CPU 不支持本项目的 fp16 训练")
    if precision in ("fp16", "bf16"):
        if not torch.amp.autocast_mode.is_autocast_available(device_type):
            raise RuntimeError(f"设备类型 {device_type} 不支持 {precision} autocast")
        if precision == "bf16" and device_type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16")

    if strategy == "none":
        context = RuntimeContext(
            device=context_device, rank=rank, local_rank=local_rank, world_size=world_size,
            enabled=False, backend=None, owns_process_group=False
        )
        return context, device_summary

    created_group = False
    try:
        dist.init_process_group(
            backend=backend, init_method="env://", rank=rank, world_size=world_size,
            timeout=timedelta(minutes=30)
        )
        created_group = True
        context = RuntimeContext(
            device=context_device, rank=rank, local_rank=local_rank, world_size=world_size,
            enabled=True, backend=backend, owns_process_group=True
        )
        return context, device_summary
    except Exception:
        if created_group and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        raise
