import torch
import re

from torch.mps import driver_allocated_memory, recommended_max_memory


def check_device(device, gpu_nums):
    if not isinstance(device, str):
        raise ValueError(f"device: {device} 必须为字符串类型的喵! 现在的type: {type(device)}")
    if not type(gpu_nums) is int:
        raise ValueError(f"gpu_nums: {gpu_nums} 必须为整型的喵! 现在的type: {type(gpu_nums)}")
    if gpu_nums <= 0:
        raise ValueError(f"gpu_nums: {gpu_nums} 必须为正数喵!")
    cuda_is_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    mps_is_available = torch.backends.mps.is_available()
    if device == "cpu":
        if not gpu_nums == 1:
            raise ValueError(f"device选择为cpu时，gpu_nums必须为1喵!")
        summary_dict = {
            "backend": "cpu",
            "requested_device": device,
            "gpu_nums": gpu_nums,
            "cuda_is_available": cuda_is_available,
            "cuda_count": cuda_count,
            "mps_is_available": mps_is_available,
            "device_index": None,
            "device_name": None,
            "total_memory": None
        }
        return summary_dict
    elif device == "mps":
        if not mps_is_available:
            raise ValueError(f"当前设备mps不可用喵!")
        if not gpu_nums == 1:
            raise ValueError(f"device选择为mps时，gpu_nums必须为1喵!")
        summary_dict = {
            "backend": "mps",
            "requested_device": device,
            "gpu_nums": gpu_nums,
            "cuda_is_available": cuda_is_available,
            "cuda_count": cuda_count,
            "mps_is_available": mps_is_available,
            "device_index": None,
            "device_name": None,
            "total_memory": None
        }
        return summary_dict
    elif re.fullmatch(r'(cuda|cuda:(\d+))$', device):
        if not cuda_is_available:
            raise ValueError(f"当前设备cuda不可用喵!")
        if not cuda_count > 0:
            raise ValueError(f"当前设备中cuda数为0喵!")
        if gpu_nums > cuda_count:
            raise ValueError(f"gpu_nums的数量不能大于cuda的数量喵! gpu_nums: {gpu_nums}, cuda_count: {cuda_count}")
        if gpu_nums == 1:
            if device == "cuda":
                device_index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device_index)
                device_name = properties.name
                total_memory = properties.total_memory
            elif re.fullmatch(r'(cuda:(\d+))$', device):
                res = re.match(r'(cuda:(\d+))$', device)
                cuda_n = int(res.group(2))
                device_index = cuda_n
                if not device_index < cuda_count:
                    raise ValueError(f"device_index: {device_index} 越界了喵! cuda_count: {cuda_count}")
                properties = torch.cuda.get_device_properties(cuda_n)
                device_name = properties.name
                total_memory = properties.total_memory
            else:
                raise ValueError(f"device: {device} 的内容不符合规范喵!")
            summary_dict = {
                "backend": "cuda",
                "requested_device": device,
                "gpu_nums": gpu_nums,
                "cuda_is_available": cuda_is_available,
                "cuda_count": cuda_count,
                "mps_is_available": mps_is_available,
                "device_index": device_index,
                "device_name": device_name,
                "total_memory": total_memory
            }
            return summary_dict
        elif gpu_nums > 1:
            devices_list = []
            if re.fullmatch(r'(cuda:(\d+))$', device):
                raise ValueError(f"目前是多卡训练模式! 不能直接指定cuda编号!")
            elif device == "cuda":
                for device_index in range(gpu_nums):
                    properties = torch.cuda.get_device_properties(device_index)
                    device_name = properties.name
                    total_memory = properties.total_memory
                    device_info = {
                        "device_index": f"cuda:{device_index}",
                        "device_name": device_name,
                        "total_memory": total_memory
                    }
                    devices_list.append(device_info)
                summary_dict = {
                    "backend": "cuda",
                    "requested_device": device,
                    "gpu_nums": gpu_nums,
                    "cuda_is_available": cuda_is_available,
                    "cuda_count": cuda_count,
                    "mps_is_available": mps_is_available,
                    "devices": devices_list,
                }
                return summary_dict
    else:
        raise ValueError(f"device: {device} 为非法设备喵! 当前框架仅支持: cpu, cuda, mps")

def get_memory_summary(device_summary):
    if not isinstance(device_summary, dict):
        raise ValueError(f"device_summary 必须为dict类型的喵! 现在的type: {type(device_summary)}")
    if not device_summary:
        raise ValueError(f"device_summary 不能为空喵!")
    if not "backend" in device_summary:
        raise ValueError(f"device_summary 中缺少了字段backend喵!")
    backend = device_summary["backend"]
    if not "gpu_nums" in device_summary:
        raise ValueError(f"device_summary 中缺少了字段gpu_nums喵!")
    gpu_nums = device_summary["gpu_nums"]
    if backend == "cpu":
        return None
    elif backend == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(f"mps现在不可用喵!")
        current_allocated_memory = torch.mps.current_allocated_memory()
        driver_allocated_memory = torch.mps.driver_allocated_memory()
        recommended_max_memory = torch.mps.recommended_max_memory()
        summary_dict = {
            "backend": backend,
            "current_allocated_memory": current_allocated_memory,
            "driver_allocated_memory": driver_allocated_memory,
            "recommended_max_memory": recommended_max_memory,
        }
        return summary_dict
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"cuda现在不可用喵!")
        if gpu_nums == 1:
            device_index = device_summary["device_index"]
            allocated_memory = torch.cuda.memory_allocated(device_index)
            reserved_memory = torch.cuda.memory_reserved(device_index)
            max_allocated_memory = torch.cuda.max_memory_allocated(device_index)
            max_reserved_memory = torch.cuda.max_memory_reserved(device_index)
            free_memory, total_memory = torch.cuda.mem_get_info(device_index)
            summary_dict = {
                "backend": "cuda",
                "gpu_nums": gpu_nums,
                "device_index": device_index,
                "allocated_memory": allocated_memory,
                "reserved_memory": reserved_memory,
                "max_allocated_memory": max_allocated_memory,
                "max_reserved_memory": max_reserved_memory,
                "free_memory": free_memory,
                "total_memory": total_memory,
            }
            return summary_dict
        elif gpu_nums > 1:
            summary_dict = {
                "backend": "cuda",
                "gpu_nums": gpu_nums,
                "devices": []
            }
            for device_index in range(gpu_nums):
                allocated_memory = torch.cuda.memory_allocated(device_index)
                reserved_memory = torch.cuda.memory_reserved(device_index)
                max_allocated_memory = torch.cuda.max_memory_allocated(device_index)
                max_reserved_memory = torch.cuda.max_memory_reserved(device_index)
                free_memory, total_memory = torch.cuda.mem_get_info(device_index)
                sub_dict = {
                    "device_index": device_index,
                    "allocated_memory": allocated_memory,
                    "reserved_memory": reserved_memory,
                    "max_allocated_memory": max_allocated_memory,
                    "max_reserved_memory": max_reserved_memory,
                    "free_memory": free_memory,
                    "total_memory": total_memory,
                }
                summary_dict["devices"].append(sub_dict)
            return summary_dict
    else:
        raise ValueError(f"收到的device_summary: {device_summary} 不符合规范喵!")

def reset_peak_memory_stats(device_summary):
    if not isinstance(device_summary, dict):
        raise ValueError(f"device_summary 必须为dict类型的喵! 现在的type: {type(device_summary)}")
    if not device_summary:
        raise ValueError(f"device_summary 不能为空喵!")
    if not "backend" in device_summary:
        raise ValueError(f"device_summary 中缺少了字段backend喵!")
    backend = device_summary["backend"]
    if not "gpu_nums" in device_summary:
        raise ValueError(f"device_summary 中缺少了字段gpu_nums喵!")
    gpu_nums = device_summary["gpu_nums"]
    if backend == "cpu":
        return None
    elif backend == "mps":
        return None
    elif backend == "cuda":
        if gpu_nums == 1:
            device_index = device_summary["device_index"]
            torch.cuda.reset_peak_memory_stats(device_index)
            return None
        elif gpu_nums > 1:
            for device_index in range(gpu_nums):
                torch.cuda.reset_peak_memory_stats(device_index)
            return None
    else:
        raise ValueError(f"收到的device_summary: {device_summary} 不符合规范喵!")

def get_memory_warning(memory_summary, warning_threshold=0.9):
    if memory_summary is None:
        return None
    else:
        backend = memory_summary["backend"]
    if backend == "mps":
        current_allocated_memory = memory_summary["current_allocated_memory"]
        driver_allocated_memory = memory_summary["driver_allocated_memory"]
        recommended_max_memory = memory_summary["recommended_max_memory"]
        if recommended_max_memory <= 0:
            usage_ratio = None
            is_warning = False
            warning_check_available = False
        else:
            usage_ratio = driver_allocated_memory / recommended_max_memory
            if usage_ratio >= warning_threshold:
                is_warning = True
                warning_check_available = True
            else:
                is_warning = False
                warning_check_available = True
        summary_dict = {
            "backend": backend,
            "current_allocated_memory": current_allocated_memory,
            "driver_allocated_memory": driver_allocated_memory,
            "recommended_max_memory": recommended_max_memory,
            "usage_ratio": usage_ratio,
            "warning_threshold": warning_threshold,
            "is_warning": is_warning,
            "warning_check_available": warning_check_available,
        }
        return summary_dict
    elif backend == "cuda":
        gpu_nums = memory_summary["gpu_nums"]
        if gpu_nums == 1:
            device_index = memory_summary["device_index"]
            allocated_memory = memory_summary["allocated_memory"]
            reserved_memory = memory_summary["reserved_memory"]
            max_allocated_memory = memory_summary["max_allocated_memory"]
            max_reserved_memory = memory_summary["max_reserved_memory"]
            free_memory = memory_summary["free_memory"]
            total_memory = memory_summary["total_memory"]
            usage_ratio = max_reserved_memory / total_memory
            if usage_ratio >= warning_threshold:
                is_warning = True
                warning_check_available = True
            else:
                is_warning = False
                warning_check_available = True
            summary_dict = {
                "backend": "cuda",
                "gpu_nums": gpu_nums,
                "device_index": device_index,
                "allocated_memory": allocated_memory,
                "reserved_memory": reserved_memory,
                "max_allocated_memory": max_allocated_memory,
                "max_reserved_memory": max_reserved_memory,
                "free_memory": free_memory,
                "total_memory": total_memory,
                "usage_ratio": usage_ratio,
                "warning_threshold": warning_threshold,
                "is_warning": is_warning,
                "warning_check_available": warning_check_available,
            }
            return summary_dict
        elif gpu_nums > 1:
            summary_dict = {
                "backend": "cuda",
                "gpu_nums": gpu_nums,
                "devices": []
            }
            for device_index in range(gpu_nums):
                sub_dict = memory_summary["devices"][device_index]
                allocated_memory = sub_dict["allocated_memory"]
                reserved_memory = sub_dict["reserved_memory"]
                max_allocated_memory = sub_dict["max_allocated_memory"]
                max_reserved_memory = sub_dict["max_reserved_memory"]
                free_memory = sub_dict["free_memory"]
                total_memory = sub_dict["total_memory"]
                usage_ratio = max_reserved_memory / total_memory
                if usage_ratio >= warning_threshold:
                    is_warning = True
                    warning_check_available = True
                else:
                    is_warning = False
                    warning_check_available = True
                sub_summary = {
                    "device_index": device_index,
                    "allocated_memory": allocated_memory,
                    "reserved_memory": reserved_memory,
                    "max_allocated_memory": max_allocated_memory,
                    "max_reserved_memory": max_reserved_memory,
                    "free_memory": free_memory,
                    "total_memory": total_memory,
                    "usage_ratio": usage_ratio,
                    "warning_threshold": warning_threshold,
                    "is_warning": is_warning,
                    "warning_check_available": warning_check_available,
                }
                summary_dict["devices"].append(sub_summary)
            return summary_dict
    else:
        raise ValueError(f"收到的memory_summary: {memory_summary} 不符合规范喵!")

def get_oom_info(error, device_summary, stage):
    backend = device_summary["backend"]
    if backend == "cpu":
        return None
    elif backend == "cuda":
        if not isinstance(error, torch.cuda.OutOfMemoryError):
            is_oom = False
            original_message = None
            cuda_advice = None
        else:
            is_oom = True
            original_message = str(error)
            cuda_advice = [f"减小dataloader.batch_size",
                           f"减小dataset.max_seq_length",
                           f"减小model.hidden_size",
                           f"减小model.intermediate_size",
                           f"减小model.num_layers",
                           f"减小micro batch，增大accumulation_steps",
                           f"检查是否有其他进程占用显存",]
        summary_dict = {
            "backend": backend,
            "is_oom": is_oom,
            "stage": stage,
            "original_message": original_message,
            "cuda_advice": cuda_advice,
        }
        if is_oom:
            return summary_dict
        else:
            return None
    elif backend == "mps":
        if "out of memory" in str(error).casefold():
            is_oom = True
            original_message = str(error)
            mps_advice = [f"减小dataloader.batch_size",
                           f"减小dataset.max_seq_length",
                           f"减小model.hidden_size",
                           f"减小model.intermediate_size",
                           f"减小model.num_layers",
                           f"减小micro batch，增大accumulation_steps",
                           f"关闭其他高内存应用",]
        else:
            is_oom = False
            original_message = None
            mps_advice = None
        summary_dict = {
            "backend": backend,
            "is_oom": is_oom,
            "stage": stage,
            "original_message": original_message,
            "mps_advice": mps_advice,
        }
        if is_oom:
            return summary_dict
        else:
            return None
    else:
        raise ValueError(f"收到的device_summary: {device_summary} 不符合规范喵!")