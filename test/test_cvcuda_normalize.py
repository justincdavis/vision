import inspect

import pytest
import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._utils import _get_kernel

cvcuda = pytest.importorskip('cvcuda', minversion='0.19.0')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')


def unpack(value):
    if isinstance(value, tuple):
        return tuple(unpack(item) for item in value)
    return F.cvcuda_to_tensor(value)


def close(actual, expected, dtype):
    if isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            close(a, e, dtype)
        return
    # U8 kernels round independently; F32 uses floating-point arithmetic.
    torch.testing.assert_close(actual, expected, atol=1 if dtype == torch.uint8 else 2e-5, rtol=0)


@pytest.mark.parametrize('name', ['normalize'])
def test_signature(name):
    actual = inspect.signature(_get_kernel(getattr(F, name), cvcuda.Tensor)).parameters
    expected = inspect.signature(getattr(F, name)).parameters
    assert actual['stream'].kind == inspect.Parameter.KEYWORD_ONLY
    assert actual['stream'].default is None
    actual = [p for n,p in actual.items() if n not in ('stream','seed')]
    def normalize(p):
        default = p.default
        if hasattr(default, 'value'):
            default = default.value
        if name == 'to_dtype' and p.name == 'dtype':
            default = 'float32'
        return p.name, p.kind, default
    assert [normalize(p) for p in actual] == [normalize(p) for p in expected.values()]


@pytest.mark.parametrize('shape', [(3,11,13),(2,3,11,13)])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_functional_and_transform(shape, dtype):
    torch.manual_seed(23)
    image = torch.rand(shape, device='cuda')
    if len(shape) == 4:
        image[0].mul_(.1)
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    for name, kwargs in [('normalize', dict(mean=[.1,.2,.3], std=[.5,.6,.7]))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.Normalize([.1,.2,.3],[.5,.6,.7])]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('mean,std', [(.2, .3), ([.2], [.3]), (None, None)])
@pytest.mark.parametrize('inplace', [False, True])
def test_normalize_broadcast_gray_and_inplace(channels, mean, std, inplace):
    if mean is None:
        mean = [.1, .2, .3][:channels]
        std = [.5, .6, .7][:channels]
    image = torch.linspace(0, 1, 2 * channels * 11 * 13, device='cuda').reshape(2, channels, 11, 13)
    original = image.clone()
    inpt = F.as_cvcuda_tensor(image)
    result = F.normalize(inpt, mean=mean, std=std, inplace=inplace)
    close(unpack(result), F.normalize(original, mean=mean, std=std), torch.float32)
    if inplace:
        assert result is inpt
        close(image, unpack(result), torch.float32)
    else:
        torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize('mean,std,error,match', [
    ([0], [0], ValueError, 'std evaluated to zero'),
    ([.1, .2], [1], RuntimeError, 'Invalid base value count'),
    ([0], [1, 2], RuntimeError, 'Invalid scale value count'),
])
def test_normalize_parameter_errors(mean, std, error, match):
    image = F.as_cvcuda_tensor(torch.ones(3, 11, 13, device='cuda'))
    with pytest.raises(error, match=match):
        F.normalize(image, mean=mean, std=std)


def test_normalize_rejects_integer_input():
    image = F.as_cvcuda_tensor(torch.ones(3, 11, 13, device='cuda', dtype=torch.uint8))
    with pytest.raises(TypeError, match='float tensor'):
        F.normalize(image, mean=[0], std=[1])


def test_normalize_half_native_reference():
    image = torch.tensor([0., .5, 1.], device='cuda', dtype=torch.float16).reshape(1, 1, 3)
    expected = (image - torch.tensor(.2, device='cuda', dtype=torch.float16)) / torch.tensor(.3, device='cuda', dtype=torch.float16)
    torch.testing.assert_close(F.normalize(image, mean=[.2], std=[.3]), expected, atol=0, rtol=0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='CV-CUDA 0.19 F16 normalize differs from TorchVision half intermediate arithmetic')
def test_normalize_half_precision_parity():
    _get_kernel(F.normalize, cvcuda.Tensor)
    image = torch.linspace(0, 1, 143, device='cuda', dtype=torch.float16).reshape(1, 11, 13)
    expected = F.normalize(image, mean=[.2], std=[.3])
    actual = unpack(F.normalize(F.as_cvcuda_tensor(image), mean=[.2], std=[.3]))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
