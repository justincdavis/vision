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


@pytest.mark.parametrize('name', ['adjust_brightness', 'adjust_contrast', 'adjust_hue', 'adjust_saturation'])
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
    for name, kwargs in [('adjust_brightness',dict(brightness_factor=.5)),('adjust_contrast',dict(contrast_factor=0.)),('adjust_contrast',dict(contrast_factor=1.5)),('adjust_hue',dict(hue_factor=.1)),('adjust_saturation',dict(saturation_factor=.5))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.ColorJitter(brightness=.2, contrast=.3, saturation=.4, hue=.1)]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
def test_contrast_means_are_per_image(dtype):
    image = torch.stack([torch.full((3,11,13),.1),torch.full((3,11,13),.8)]).cuda()
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    actual = unpack(F.adjust_contrast(F.as_cvcuda_tensor(image), contrast_factor=0))
    close(actual, F.adjust_contrast(image, contrast_factor=0), dtype)
    assert actual[1].float().mean() > 5 * actual[0].float().mean()


@pytest.mark.parametrize('name,kwargs', [
    ('adjust_brightness', {'brightness_factor':-.1}),
    ('adjust_contrast', {'contrast_factor':-.1}),
    ('adjust_saturation', {'saturation_factor':-.1}),
    ('adjust_hue', {'hue_factor':.6}),
])
def test_color_invalid_factors(name, kwargs):
    image = F.as_cvcuda_tensor(torch.rand(3,11,13,device='cuda'))
    with pytest.raises(ValueError):
        getattr(F,name)(image, **kwargs)
