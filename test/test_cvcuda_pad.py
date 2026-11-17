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


@pytest.mark.parametrize('name', ['pad'])
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
    for name, kwargs in [('pad', dict(padding=[1,2,3,4],fill=7)),('pad',dict(padding=[2,3],padding_mode='reflect')),('pad',dict(padding=[2,3],padding_mode='edge')),('pad',dict(padding=[2,3],padding_mode='symmetric'))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.Pad((1,2,3,4),fill=7)]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('shape', [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('padding', [
    [1, 2, 3, 4],
    pytest.param([-2, -3, -1, -2], marks=pytest.mark.xfail(
        strict=True, raises=AssertionError, reason='Released pad passes negative borders to copymakeborder'
    )),
    pytest.param([-2, 3, 4, -1], marks=pytest.mark.xfail(
        strict=True, raises=AssertionError, reason='Released pad passes negative borders to copymakeborder'
    )),
])
@pytest.mark.parametrize('fill', [7, [7], [3, 5, 7]])
def test_pad_padding_and_fill(shape, dtype, padding, fill):
    image = torch.arange(torch.tensor(shape).prod().item(), device='cuda').reshape(shape).remainder(251).to(dtype)
    kwargs = dict(padding=padding, fill=fill)
    expected = F.pad(image, **kwargs)
    try:
        actual = unpack(F.pad(F.as_cvcuda_tensor(image), **kwargs))
    except RuntimeError as error:
        if 'top >= 0 && bottom >= 0 && left >= 0 && right >= 0' not in str(error):
            raise
        raise AssertionError('Known released negative-border rejection') from error
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(unpack(v2.Pad(padding, fill=fill)(F.as_cvcuda_tensor(image))), expected, atol=0, rtol=0)


def test_pad_negative_native_reference_and_released_error_contract():
    image = torch.arange(3 * 11 * 13, device='cuda').reshape(3, 11, 13).float()
    kwargs = dict(padding=[-2, -3, -1, -2], fill=7)
    expected = F.pad(image, **kwargs)
    assert expected.shape == (3, 6, 10)
    torch.testing.assert_close(expected, image[:, 3:-2, 2:-1], atol=0, rtol=0)
    with pytest.raises(RuntimeError) as caught:
        cvcuda.tv.pad(F.as_cvcuda_tensor(image), **kwargs)
    assert 'top >= 0 && bottom >= 0 && left >= 0 && right >= 0' in str(caught.value)
