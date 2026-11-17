import functools
from typing import Any

import torch
from torch.torch_version import TorchVersion

from ._utils import _register_kernel_internal

try:
    import cvcuda
except ModuleNotFoundError as error:
    if error.name != "cvcuda":
        raise
    cvcuda = None

if cvcuda is not None and TorchVersion(getattr(cvcuda, "__version__", "0")) >= "0.19.0":
    from cvcuda import tv
else:
    cvcuda = tv = None


def _is_cvcuda_available() -> bool:
    return cvcuda is not None


def _import_cvcuda():
    if cvcuda is None:
        raise ImportError("CV-CUDA >= 0.19.0 with cvcuda.tv is required for CV-CUDA transforms.")
    return cvcuda


def _is_cvcuda_tensor(inpt: Any) -> bool:
    return cvcuda is not None and isinstance(inpt, cvcuda.Tensor)


def _validate_cvcuda_image(inpt) -> None:
    if (str(inpt.layout), inpt.ndim) not in (("CHW", 3), ("NCHW", 4)):
        raise ValueError("CV-CUDA images must have CHW or NCHW layout and rank 3 or 4.")


def _register_cvcuda_kernel(functional, kernel):
    """Register a cvcuda.tv kernel on PyTorch's current stream, without changing Torch dispatch."""

    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        _validate_cvcuda_image(inpt)
        device = inpt.cuda().__dlpack_device__()[1]
        with torch.cuda.device(device):
            stream = cvcuda.as_stream(torch.cuda.current_stream(device).cuda_stream)
            return kernel(inpt, *args, **kwargs, stream=stream)

    _register_kernel_internal(functional, _import_cvcuda().Tensor)(wrapper)
    return wrapper
