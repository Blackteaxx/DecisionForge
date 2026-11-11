"""Probe CUDA in a subprocess so the web server never imports torch before Unsloth."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=4)
def _cuda_probe(interpreter, versions):
    code = """
import json, torch
result = {'cuda_available': torch.cuda.is_available()}
if result['cuda_available']:
    value = torch.ones((2, 2), device='cuda')
    assert (value @ value).sum().item() == 8
    result.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [interpreter, "-c", code],
            capture_output=True,
            text=True,
            timeout=45,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode:
            return {
                "cuda_available": False,
                "error": "CUDA/PyTorch 初始化失败，请检查驱动和训练依赖。",
            }
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {"cuda_available": False, "error": "CUDA 环境检查未完成，请检查驱动或重启工作台。"}


def capabilities():
    system = platform.system()
    missing = [
        name for name in ("torch", "unsloth", "triton") if importlib.util.find_spec(name) is None
    ]
    result = {"system": system, "gpu_runtime": False, "missing_packages": missing}
    if system not in {"Linux", "Windows"}:
        result["note"] = "训练需要 Windows 或 Linux 的 NVIDIA CUDA 环境。"
    elif missing:
        result["note"] = (
            "缺少训练依赖：" + "、".join(missing) + "。请运行 uv sync --extra train-unsloth。"
        )
    else:
        versions = []
        for name in ("torch", "unsloth", "triton-windows" if system == "Windows" else "triton"):
            try:
                versions.append(version(name))
            except PackageNotFoundError:
                versions.append(None)
        probe = _cuda_probe(sys.executable, tuple(versions))
        result.update(probe)
        result["gpu_runtime"] = bool(probe.get("cuda_available"))
        result["note"] = (
            f"已检测到 {system} CUDA 训练环境：{probe['gpu']}。"
            if result["gpu_runtime"]
            else probe.get("error", "CUDA 不可用，请检查 NVIDIA 驱动与 CUDA 版 PyTorch。")
        )
    return result
