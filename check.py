import platform
import sys

import matplotlib
import numpy
import pytest
import torch
import tqdm
import yaml


def main():
    print("=== NLP 环境自查 ===")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"开发平台: {platform.platform()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {numpy.__version__}")
    print(f"PyYAML版本: {yaml.__version__}")
    print(f"tqdm版本: {tqdm.__version__}")
    print(f"Matplotlib版本: {matplotlib.__version__}")
    print(f"pytest版本: {pytest.__version__}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA可用: {cuda_available}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    if not cuda_available:
        print("未检测到可用 CUDA，请查看pytorch是否下载为GPU版本")
        return

    print(f"GPU 数量: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(f"GPU {index}: {properties.name} ({memory_gib:.1f} GiB)")

    try:
        left = torch.randn(512, 512, device="cuda")
        right = torch.randn(512, 512, device="cuda")
        result = left @ right
        torch.cuda.synchronize()
        print(f"CUDA computation: passed ({result.device})")
    except RuntimeError as error:
        raise SystemExit(f"CUDA computation failed: {error}") from error


if __name__ == "__main__":
    main()
