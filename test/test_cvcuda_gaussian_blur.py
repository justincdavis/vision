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


@pytest.mark.parametrize('name', ['gaussian_blur'])
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
@pytest.mark.parametrize('dtype', [torch.uint8,torch.float32])
def test_functional_and_transform(shape, dtype):
    torch.manual_seed(23)
    image = torch.rand(shape, device='cuda')
    if len(shape) == 4:
        image[0].mul_(.1)
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    for name, kwargs in [('gaussian_blur', dict(kernel_size=[3,5], sigma=[.8,1.2]))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.GaussianBlur(3, sigma=(.8,.8))]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('kernel_size,sigma', [(3, None), ([3, 5], None), (3, [.7])])
def test_gaussian_blur_gray(dtype, kernel_size, sigma):
    image = torch.arange(2 * 11 * 13, device='cuda').reshape(2, 1, 11, 13).remainder(127).to(dtype)
    if dtype == torch.float32:
        image.div_(127)
    kwargs = dict(kernel_size=kernel_size, sigma=sigma)
    close(unpack(F.gaussian_blur(F.as_cvcuda_tensor(image), **kwargs)), F.gaussian_blur(image, **kwargs), dtype)


@pytest.mark.parametrize('kernel_size,sigma,error,match', [
    ([2, 3], None, ValueError, 'odd positive'),
    ([0, 3], None, ValueError, 'odd positive'),
    ([3], None, ValueError, 'two integers'),
    (True, None, ValueError, 'two integers'),
    (3.5, None, ValueError, 'two integers'),
    (3, 0, ValueError, 'positive values'),
    (3, [-1, 1], ValueError, 'positive values'),
    (3, [1, 2, 3], ValueError, 'length should be 2'),
    (3, 'bad', TypeError, 'sigma should be'),
])
def test_gaussian_blur_parameter_errors(kernel_size, sigma, error, match):
    image = F.as_cvcuda_tensor(torch.ones(1, 11, 13, device='cuda'))
    with pytest.raises(error, match=match):
        F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
