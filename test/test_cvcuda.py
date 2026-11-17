import gc
import inspect
import pathlib
import subprocess
import sys

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._cvcuda import _is_cvcuda_available


@pytest.mark.parametrize("version", [None, "0.18.0"])
def test_optional_dependency(version):
    helper = pathlib.Path(F.__file__).with_name("_cvcuda.py")
    script = f"""
import importlib.abc
import sys
import types
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cvcuda':
            raise ModuleNotFoundError("No module named 'cvcuda'", name='cvcuda')
if {version!r} is None:
    sys.meta_path.insert(0, Block())
else:
    module = types.ModuleType('cvcuda')
    module.__version__ = {version!r}
    sys.modules['cvcuda'] = module
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import torch
assert torch.equal(F.horizontal_flip(torch.arange(12).reshape(1, 3, 4)), torch.arange(12).reshape(1, 3, 4).flip(-1))
try:
    F.to_cvcuda_tensor(torch.zeros(3, 4, 5))
except ImportError as error:
    assert '0.19.0' in str(error)
else:
    raise AssertionError('Expected ImportError')
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=helper.parents[4])


@pytest.mark.parametrize("error", ["ImportError('broken dependency')", "ModuleNotFoundError('missing transitive', name='nvcv')"])
def test_broken_dependency_not_hidden(error):
    script = f"""
import importlib.abc
import sys
class Broken(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cvcuda':
            raise {error}
sys.meta_path.insert(0, Broken())
try:
    from torchvision.transforms import v2
except ImportError as error:
    assert 'broken dependency' in str(error) or 'missing transitive' in str(error)
else:
    raise AssertionError('Broken dependency was hidden')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


cuda = pytest.mark.skipif(
    not (_is_cvcuda_available() and torch.cuda.is_available()), reason="Requires CV-CUDA >= 0.19.0 and CUDA"
)


@cuda
@pytest.mark.parametrize("shape", [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_conversion(shape, dtype):
    image = torch.arange(torch.tensor(shape).prod().item(), device="cuda").reshape(shape).to(dtype)
    wrapped = F.as_cvcuda_tensor(image)
    assert str(wrapped.layout) == ("CHW" if len(shape) == 3 else "NCHW")
    result = F.cvcuda_to_tensor(wrapped)
    assert result.data_ptr() == image.data_ptr()
    assert result.device == image.device
    assert result.dtype == dtype
    torch.testing.assert_close(result, image)
    del image, wrapped
    gc.collect()
    result.fill_(7)
    assert (result == 7).all()


@cuda
def test_copy_and_transform_conversion():
    image = torch.arange(3 * 11 * 13, dtype=torch.float32).reshape(3, 11, 13).transpose(-1, -2)
    wrapped = v2.ToCVCUDATensor()(image)
    result = v2.CVCUDAToTensor()(wrapped)
    torch.testing.assert_close(result.cpu(), image)
    assert result.is_contiguous()
    with pytest.raises(ValueError, match="contiguous CUDA"):
        F.as_cvcuda_tensor(image)
    with pytest.raises(ValueError, match="contiguous CUDA"):
        F.as_cvcuda_tensor(image.cuda())
    with pytest.raises(ValueError, match="CUDA device"):
        F.to_cvcuda_tensor(image, device="cpu")
    torch.testing.assert_close(F.cvcuda_to_tensor(v2.ToCVCUDATensor()(tv_tensors.Image(image))).cpu(), image)


@pytest.mark.parametrize("converter", [F.as_cvcuda_tensor, F.to_cvcuda_tensor])
def test_input_validation(converter):
    with pytest.raises(TypeError, match="torch.Tensor"):
        converter(object())
    for rank in (2, 5):
        with pytest.raises(ValueError, match="rank 3 or 4"):
            converter(torch.zeros((2,) * rank))
    with pytest.raises(ValueError, match="autograd"):
        converter(torch.zeros(3, 4, 5, requires_grad=True))


@cuda
@pytest.mark.parametrize("shape", [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize(
    "functional,transform,dim",
    [(F.horizontal_flip, v2.RandomHorizontalFlip, -1), (F.vertical_flip, v2.RandomVerticalFlip, -2)],
)
def test_flip_and_metadata(shape, functional, transform, dim):
    import cvcuda

    image = torch.rand(shape, device="cuda")
    wrapped = F.as_cvcuda_tensor(image)
    assert F.get_dimensions(wrapped) == list(shape[-3:])
    assert F.get_num_channels(wrapped) == 3
    assert F.get_size(wrapped) == list(shape[-2:])
    assert v2.query_chw([wrapped]) == shape[-3:]
    assert v2.query_size([wrapped]) == shape[-2:]
    assert isinstance(functional(wrapped), cvcuda.Tensor)
    torch.testing.assert_close(F.cvcuda_to_tensor(functional(wrapped)), image.flip(dim))
    torch.testing.assert_close(F.cvcuda_to_tensor(transform(p=1)(wrapped)), image.flip(dim))
    assert transform(p=0)(wrapped) is wrapped
    labels = torch.tensor([1, 2])
    output = transform(p=1)({"labels": labels, "image": wrapped})
    assert output["labels"] is labels
    torch.testing.assert_close(F.cvcuda_to_tensor(output["image"]), image.flip(dim))
    with pytest.raises(ValueError, match="multiple"):
        v2.query_size([wrapped, F.to_cvcuda_tensor(torch.zeros(3, 5, 7))])


@cuda
@pytest.mark.parametrize("layout,shape", [("HWC", (11, 13, 3)), ("NHWC", (2, 11, 13, 3))])
def test_reject_interleaved(layout, shape):
    import cvcuda

    wrapped = cvcuda.as_tensor(torch.zeros(shape, device="cuda"), layout)
    for operation in (F.cvcuda_to_tensor, F.get_dimensions, F.get_num_channels, F.get_size, F.horizontal_flip):
        with pytest.raises(ValueError, match="CHW or NCHW"):
            operation(wrapped)


@cuda
def test_nondefault_stream_chain():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        image = torch.arange(3 * 128 * 129, device="cuda", dtype=torch.float32).reshape(3, 128, 129)
        image.mul_(0.5)
        expected = image.flip(-1).flip(-2)
        wrapped = F.as_cvcuda_tensor(image)
        del image
        gc.collect()
        result = F.cvcuda_to_tensor(F.vertical_flip(F.horizontal_flip(wrapped)))
        result.add_(1)
        expected.add_(1)
    stream.synchronize()
    torch.testing.assert_close(result, expected)


@cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_noncurrent_device():
    with torch.cuda.device(0):
        image = torch.rand(3, 11, 13, device="cuda:1")
        result = F.cvcuda_to_tensor(F.horizontal_flip(F.as_cvcuda_tensor(image)))
        assert result.device == image.device
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(result, image.flip(-1))


def test_torch_dispatch_unchanged():
    image = torch.rand(3, 11, 13)
    assert isinstance(F.horizontal_flip(image), torch.Tensor)
    torch.testing.assert_close(F.horizontal_flip(image), image.flip(-1))
    torch.testing.assert_close(v2.RandomVerticalFlip(p=1)(image), image.flip(-2))


@cuda
@pytest.mark.parametrize("functional", [F.horizontal_flip, F.vertical_flip])
def test_cvcuda_kernel_signature(functional):
    import cvcuda
    from torchvision.transforms.v2.functional._utils import _get_kernel

    kernel = _get_kernel(functional, cvcuda.Tensor)
    expected = list(inspect.signature(functional).parameters.values())
    actual = list(inspect.signature(kernel).parameters.values())
    assert actual[-1].name == "stream"
    assert actual[-1].kind == inspect.Parameter.KEYWORD_ONLY
    assert actual[-1].default is None
    assert [(p.name, p.kind, p.default) for p in actual[:-1]] == [
        (p.name, p.kind, p.default) for p in expected
    ]


@cuda
def test_unsupported_cvcuda_transform_raises():
    class UnsupportedTransform(v2.Transform):
        def transform(self, inpt, params):
            return self._call_kernel(F.uniform_temporal_subsample, inpt, 1)

    image = F.to_cvcuda_tensor(torch.zeros(3, 11, 13))
    with pytest.raises(TypeError, match="uniform_temporal_subsample"):
        UnsupportedTransform()(image)
    with pytest.raises(TypeError, match="uniform_temporal_subsample"):
        F.uniform_temporal_subsample(image, 1)


@cuda
def test_to_dtype_discovers_cvcuda_image(monkeypatch):
    image = F.to_cvcuda_tensor(torch.full((3, 4, 5), 255, dtype=torch.uint8))
    transform = v2.ToDtype(torch.float32, scale=True)
    result = object()

    def kernel(functional, inpt, *, dtype, scale):
        assert functional is F.to_dtype and inpt is image
        assert dtype is torch.float32 and scale is True
        return result

    monkeypatch.setattr(transform, "_call_kernel", kernel)
    assert transform(image) is result
