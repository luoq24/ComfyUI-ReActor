import sys
import os

repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)

# ---------------------------------------------------------------------------
# ONNX Runtime CUDA DLL 保险：
# onnxruntime-gpu 加载 CUDA/cuDNN DLL 时依赖系统 PATH。torch 自带的
# torch/lib 目录里就有匹配版本的 cudnn/cublas/cufft 等 DLL，但 ComfyUI
# 进程的 PATH 通常不含它，导致 CUDAExecutionProvider 静默回退 CPU
# （换脸耗时 10~20 倍）。这里在任何 onnxruntime 导入前把 torch/lib
# 注册进 DLL 搜索路径，彻底避免该问题。
# ---------------------------------------------------------------------------
try:
    import torch as _torch_preload
    _torch_lib = os.path.join(os.path.dirname(_torch_preload.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            import onnxruntime as _ort_check
            if "CUDAExecutionProvider" not in _ort_check.get_available_providers():
                print("[ReActor] onnxruntime CUDA provider unavailable — "
                      "check that onnxruntime-gpu is installed (not CPU onnxruntime). "
                      "See fix_onnx_gpu.bat")
        except Exception:
            pass
except Exception:
    pass
original_modules = sys.modules.copy()

# Place aside existing modules if using a1111 web ui
modules_used = [
    "modules",
    "modules.images",
    "modules.processing",
    "modules.scripts_postprocessing",
    "modules.scripts",
    "modules.shared",
]
original_webui_modules = {}
for module in modules_used:
    if module in sys.modules:
        original_webui_modules[module] = sys.modules.pop(module)

# Proceed with node setup
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Clean up imports
# Remove repo directory from path
sys.path.remove(repo_dir)
# Remove any new modules
modules_to_remove = []
for module in sys.modules:
    if module not in original_modules and not module.startswith("google.protobuf") and not module.startswith("onnx") and not module.startswith("cv2"):
        modules_to_remove.append(module)
for module in modules_to_remove:
    del sys.modules[module]

# Restore original modules
sys.modules.update(original_webui_modules)
