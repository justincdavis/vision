import inspect

import pytest
import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._utils import _get_kernel

cvcuda = pytest.importorskip("cvcuda", minversion="0.19.0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


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


@pytest.mark.parametrize('name', ['gaussian_noise'])
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
    for name, kwargs in [('gaussian_noise',dict(mean=.1,sigma=0.,clip=True))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.GaussianNoise(mean=.1,sigma=0)]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('channels', [1, 3])
def test_gaussian_noise_seed_reproducibility(dtype, channels):
    image = torch.full((2, channels, 19, 23), 127 if dtype == torch.uint8 else .5, device='cuda', dtype=dtype)
    inpt = F.as_cvcuda_tensor(image)
    kernel = _get_kernel(F.gaussian_noise, cvcuda.Tensor)
    first = unpack(kernel(inpt, sigma=.2, seed=123))
    second = unpack(kernel(inpt, sigma=.2, seed=123))
    other = unpack(kernel(inpt, sigma=.2, seed=124))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other)
    assert not torch.equal(first, image)
    assert first.min() >= 0
    assert first.max() <= (255 if dtype == torch.uint8 else 1)


@pytest.mark.parametrize('mean', [-2., 2.])
def test_gaussian_noise_clip_float(mean):
    image = torch.full((2, 1, 11, 13), .5, device='cuda')
    inpt = F.as_cvcuda_tensor(image)
    unclipped = unpack(F.gaussian_noise(inpt, mean=mean, sigma=0, clip=False))
    clipped = unpack(F.gaussian_noise(inpt, mean=mean, sigma=0, clip=True))
    close(unclipped, image + mean, torch.float32)
    close(clipped, (image + mean).clamp(0, 1), torch.float32)


def test_gaussian_noise_negative_sigma():
    image = F.as_cvcuda_tensor(torch.ones(1, 11, 13, device='cuda'))
    with pytest.raises(ValueError, match="sigma shouldn't be negative"):
        F.gaussian_noise(image, sigma=-.1)


@pytest.mark.parametrize('seed,error,match', [
    (True, TypeError, '64-bit integer'), (1.2, TypeError, '64-bit integer'),
    (-1, ValueError, 'range'), (2**64, ValueError, 'range'),
])
def test_gaussian_noise_invalid_seed(seed, error, match):
    image = F.as_cvcuda_tensor(torch.ones(1, 11, 13, device='cuda'))
    with pytest.raises(error, match=match):
        _get_kernel(F.gaussian_noise, cvcuda.Tensor)(image, seed=seed)


def test_noise_transform_distribution():
    image = torch.full((2, 3, 64, 64), .5, device='cuda')
    actual = unpack(v2.GaussianNoise(mean=.1, sigma=.05, clip=False)(F.as_cvcuda_tensor(image)))
    noise = actual - image
    assert abs(noise.mean().item() - .1) < .002
    assert abs(noise.std().item() - .05) < .002
    assert not torch.equal(actual[0], actual[1])
