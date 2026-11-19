import functools
import inspect
from typing import Any

import torch
from torch.torch_version import TorchVersion

from ._utils import _register_kernel_internal

@functools.lru_cache(maxsize=1)
def _is_cvcuda_available() -> bool:
    try:
        import cvcuda
    except ModuleNotFoundError as error:
        if error.name != "cvcuda":
            raise
        return False
    return TorchVersion(getattr(cvcuda, "__version__", "0")) >= "0.19.0"


def _import_cvcuda():
    if not _is_cvcuda_available():
        raise ImportError("CV-CUDA >= 0.19.0 with cvcuda.tv is required for CV-CUDA transforms.")
    import cvcuda

    return cvcuda


def _is_cvcuda_tensor(inpt: Any) -> bool:
    return _is_cvcuda_available() and isinstance(inpt, _import_cvcuda().Tensor)


def _validate_cvcuda_image(inpt) -> None:
    if (str(inpt.layout), inpt.ndim) not in (("CHW", 3), ("HWC", 3), ("NCHW", 4), ("NHWC", 4)):
        raise ValueError("CV-CUDA images must have CHW, HWC, NCHW, or NHWC layout and matching rank 3 or 4.")


def _register_cvcuda_kernel(functional, kernel):
    """Register a cvcuda.tv kernel on PyTorch's current stream, without changing Torch dispatch."""

    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        _validate_cvcuda_image(inpt)
        device = inpt.cuda().__dlpack_device__()[1]
        with torch.cuda.device(device):
            stream = _import_cvcuda().as_stream(torch.cuda.current_stream(device).cuda_stream)
            return kernel(inpt, *args, **kwargs, stream=stream)

    _register_kernel_internal(functional, _import_cvcuda().Tensor)(wrapper)
    return wrapper


def _register_all_cvcuda_kernels():
    if not _is_cvcuda_available():
        return

    from torchvision.transforms.v2 import functional as F

    for name in _import_cvcuda().tv.__all__:
        if name == "Compose":
            continue
        kernel = getattr(_import_cvcuda().tv, name)
        functional = getattr(F, name, None)
        if inspect.isfunction(kernel) and inspect.isfunction(functional):
            _register_cvcuda_kernel(functional, kernel)

    for name in ("get_dimensions", "get_num_channels", "get_size"):
        kernel = getattr(F._meta, f"_{name}_image_cvcuda")
        _register_kernel_internal(getattr(F, name), _import_cvcuda().Tensor)(kernel)
