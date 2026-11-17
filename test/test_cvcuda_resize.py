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


@pytest.mark.parametrize('name', ['resize'])
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
    for name, kwargs in [('resize',dict(size=[11,13])),('resize',dict(size=[17,19],interpolation=InterpolationMode.NEAREST)),('resize',dict(size=[17,19],interpolation=InterpolationMode.BILINEAR,antialias=False)),('resize',dict(size=11,max_size=13))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.Resize((7,9)),v2.Resize(7,max_size=9),v2.Compose([v2.Resize((17,19)),v2.Resize((7,9))])]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('interpolation', [InterpolationMode.NEAREST_EXACT, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
@pytest.mark.parametrize('antialias', [False, True])
def test_resize_downsampling(dtype, interpolation, antialias):
    torch.manual_seed(28)
    image = torch.rand(2, 3, 23, 29, device='cuda')
    image[0].mul_(.1)
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    kwargs = dict(size=[7, 9], interpolation=interpolation, antialias=antialias)
    actual = unpack(F.resize(F.as_cvcuda_tensor(image), **kwargs))
    close(actual, F.resize(image, **kwargs), dtype)


@pytest.mark.parametrize('kwargs,error,match', [
    (dict(size=[0, 3]), ValueError, 'positive int32'),
    (dict(size=[3, 4], max_size=9), ValueError, 'max_size should only'),
    (dict(size=9, max_size=9), ValueError, 'strictly greater'),
    (dict(size=None), ValueError, 'max_size must be an integer'),
    (dict(size=[3, 4], interpolation=InterpolationMode.LANCZOS), NotImplementedError, 'not supported'),
])
def test_resize_parameter_errors(kwargs, error, match):
    image = F.as_cvcuda_tensor(torch.ones(3, 11, 13, device='cuda'))
    with pytest.raises(error, match=match):
        F.resize(image, **kwargs)
