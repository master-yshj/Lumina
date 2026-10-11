from __future__ import annotations

import platform
import sys


def log(msg: str) -> None:
    print(msg, flush=True)


def check_python() -> bool:
    log("=== Python ===")
    log(f"版本: {sys.version.split()[0]}")
    log(f"平台: {platform.platform()}")
    if sys.version_info < (3, 10):
        log("[失败] 需要 Python >= 3.10")
        return False
    log("[通过] Python 版本满足要求")
    return True


def check_packages() -> bool:
    log("\n=== 依赖包 ===")
    ok = True
    required = {
        "torch": None,
        "yaml": "pyyaml",
        "numpy": None,
        "tqdm": None,
        "matplotlib": None,
        "plotly": None,
        "pytest": None,
    }
    for mod_name, pip_name in required.items():
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            log(f"[通过] {mod_name} {ver}")
        except Exception:
            name = pip_name or mod_name
            log(f"[失败] 缺少 {mod_name}，请安装 {name}")
            ok = False
    return ok


def check_cuda() -> bool:
    log("\n=== CUDA ===")
    try:
        import torch
    except Exception as e:
        log(f"[失败] torch 不可用: {e}")
        return False

    log(f"torch: {torch.__version__}")
    available = torch.cuda.is_available()
    log(f"CUDA 可用: {available}")
    if not available:
        log("[提示] 无可用 CUDA，CPU 仍可跑小配置，不阻塞")
        return True

    log(f"CUDA runtime: {torch.version.cuda}")
    count = torch.cuda.device_count()
    log(f"GPU 数量: {count}")
    for i in range(count):
        p = torch.cuda.get_device_properties(i)
        log(f"  GPU {i}: {p.name} {p.total_memory / 1024 ** 3:.1f}GiB")

    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        log(f"[通过] CUDA 计算正常: {c.device} {tuple(c.shape)}")
    except Exception as e:
        log(f"[失败] CUDA 计算失败: {e}")
        return False
    return True


def main() -> int:
    log("=== NLP 环境自查 ===")
    results = [
        check_python(),
        check_packages(),
        check_cuda(),
    ]
    log("\n=== 汇总 ===")
    if all(results):
        log("全部通过")
        return 0
    log("存在失败项，见上方 [失败]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())