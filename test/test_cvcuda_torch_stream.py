import gc

import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional import _cvcuda

cvcuda = pytest.importorskip("cvcuda", minversion="0.19.0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("shape", [(3, 127, 129), (2, 3, 127, 129)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("nondefault", [False, True])
def test_wrapper_uses_torch_current_stream(monkeypatch, shape, dtype, nondefault):
    calls = []
    original = cvcuda.as_stream

    def as_stream(handle):
        calls.append(handle)
        return original(handle)

    monkeypatch.setattr(_cvcuda.cvcuda, "as_stream", as_stream)
    stream = torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
    # An unrelated CV-CUDA ambient stream must not select where TorchVision runs.
    with cvcuda.Stream(), torch.cuda.stream(stream):
        image = torch.arange(torch.tensor(shape).prod().item(), device="cuda").reshape(shape).to(dtype)
        image.add_(1)
        expected = image.flip(-1).flip(-2)
        wrapped = F.as_cvcuda_tensor(image)
        del image
        gc.collect()
        result = F.cvcuda_to_tensor(F.vertical_flip(F.horizontal_flip(wrapped)))
        result.add_(2)
        expected.add_(2)
    stream.synchronize()
    assert calls == [stream.cuda_stream, stream.cuda_stream]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 127, 129), (2, 3, 127, 129)])
def test_copy_and_composed_transforms_on_nondefault_stream(shape):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        image = torch.rand(shape, device="cuda").transpose(-1, -2)
        expected = image.flip(-1).flip(-2)
        pipeline = v2.Compose([
            v2.ToCVCUDATensor(), v2.RandomHorizontalFlip(p=1),
            v2.RandomVerticalFlip(p=1), v2.CVCUDAToTensor(),
        ])
        result = pipeline(image)
        del image
        gc.collect()
        result.mul_(.5)
        expected.mul_(.5)
    stream.synchronize()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
