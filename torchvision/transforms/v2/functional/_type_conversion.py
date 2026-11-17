from types import SimpleNamespace
from typing import Union

import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import functional as _F

from ._cvcuda import _import_cvcuda, _validate_cvcuda_image


@torch.jit.unused
def to_image(inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, PIL.Image.Image):
        output = pil_to_tensor(inpt)
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(inpt)} instead."
        )
    return tv_tensors.Image(output)


to_pil_image = _F.to_pil_image
pil_to_tensor = _F.pil_to_tensor


def _validate_cvcuda_input(inpt):
    if not isinstance(inpt, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(inpt)}.")
    if inpt.ndim not in (3, 4):
        raise ValueError("Input must have CHW or NCHW shape (rank 3 or 4).")
    if inpt.requires_grad:
        raise ValueError("CV-CUDA does not support autograd. Detach the input explicitly.")


@torch.jit.unused
def as_cvcuda_tensor(inpt: torch.Tensor):
    """Wrap a contiguous CUDA tensor without copying, preserving CHW/NCHW layout and device.

    The returned tensor shares storage with the input. CV-CUDA does not support autograd.
    """
    _validate_cvcuda_input(inpt)
    cvcuda = _import_cvcuda()
    if inpt.device.type != "cuda" or not inpt.is_contiguous():
        raise ValueError("as_cvcuda_tensor requires a contiguous CUDA tensor. Use to_cvcuda_tensor to copy.")
    interface = dict(inpt.__cuda_array_interface__)
    interface.update(version=3, stream=torch.cuda.current_stream(inpt.device).cuda_stream or -1)
    owner = SimpleNamespace(_tensor=inpt, __cuda_array_interface__=interface)
    with torch.cuda.device(inpt.device):
        return cvcuda.as_tensor(owner, "CHW" if inpt.ndim == 3 else "NCHW")


@torch.jit.unused
def to_cvcuda_tensor(inpt: torch.Tensor, *, device=None):
    """Convert a CHW/NCHW Torch tensor to CV-CUDA without scaling or permuting values.

    CPU inputs move to the current CUDA device; CUDA inputs keep their device unless
    ``device`` is specified. Non-contiguous inputs are copied. Otherwise storage is shared.
    """
    _validate_cvcuda_input(inpt)
    _import_cvcuda()
    device = torch.device(device) if device is not None else (inpt.device if inpt.is_cuda else torch.device("cuda"))
    if device.type != "cuda":
        raise ValueError("CV-CUDA tensors require a CUDA device.")
    return as_cvcuda_tensor(inpt.to(device=device).contiguous())


@torch.jit.unused
def cvcuda_to_tensor(inpt) -> torch.Tensor:
    """Return a zero-copy CUDA Torch tensor, preserving CHW/NCHW layout and device."""
    cvcuda = _import_cvcuda()
    if not isinstance(inpt, cvcuda.Tensor):
        raise TypeError(f"Input must be a cvcuda.Tensor, got {type(inpt)}.")
    _validate_cvcuda_image(inpt)
    buffer = inpt.cuda()
    device = torch.device("cuda", buffer.__dlpack_device__()[1])
    with torch.cuda.device(device):
        return torch.as_tensor(buffer, device=device)
