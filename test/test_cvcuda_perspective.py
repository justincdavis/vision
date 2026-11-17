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


@pytest.mark.parametrize('name', ['perspective'])
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
def test_functional(shape, dtype):
    torch.manual_seed(23)
    image = torch.rand(shape, device='cuda')
    if len(shape) == 4:
        image[0].mul_(.1)
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    for name, kwargs in [('perspective',dict(startpoints=[[0,0],[12,0],[12,10],[0,10]],endpoints=[[0,0],[12,0],[12,10],[0,10]],fill=7)),('perspective',dict(startpoints=None,endpoints=None,coefficients=[1.,0.,2.,0.,1.,3.,0.,0.],fill=7))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
def test_perspective_nontrivial(dtype):
    torch.manual_seed(28)
    image = torch.rand(2, 3, 23, 29, device='cuda')
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    kwargs = dict(startpoints=[[0, 0], [28, 0], [28, 22], [0, 22]], endpoints=[[2, 1], [26, 2], [25, 20], [1, 21]], fill=7 if dtype == torch.uint8 else .2)
    close(unpack(F.perspective(F.as_cvcuda_tensor(image), **kwargs)), F.perspective(image, **kwargs), dtype)


@pytest.mark.parametrize('shape', [(3, 11, 13), (2, 3, 11, 13), (3, 23, 29), (2, 3, 23, 29)])
@pytest.mark.parametrize('dtype,fill', [
    (torch.uint8, 7),
    (torch.float32, .2),
    pytest.param(torch.float32, 7, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason='Released cvcuda.tv FP32 high-fill perspective exceeds 2e-5 (up to 4.02e-5)'
    )),
])
def test_random_perspective(shape, dtype, fill):
    torch.manual_seed(23)
    image = torch.rand(shape, device='cuda')
    if len(shape) == 4:
        image[0].mul_(.1)
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    transform = v2.RandomPerspective(distortion_scale=.3, p=1, fill=fill)
    torch.manual_seed(17)
    expected = transform(image.clone())
    torch.manual_seed(17)
    actual = transform(F.as_cvcuda_tensor(image.clone()))
    assert isinstance(actual, cvcuda.Tensor)
    close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('shape', [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize('coefficients', [[1., 0., 0., 0., 1., 0., 0., 0.], [1., 0., 2., 0., 1., 3., 0., 0.]])
def test_perspective_identity_and_integer_translation_exact(dtype, shape, coefficients):
    image = torch.arange(torch.tensor(shape).prod().item(), device='cuda').reshape(shape).remainder(251).to(dtype)
    kwargs = dict(startpoints=None, endpoints=None, coefficients=coefficients,
                  interpolation=InterpolationMode.NEAREST, fill=7)
    expected = F.perspective(image, **kwargs)
    actual = unpack(F.perspective(F.as_cvcuda_tensor(image), **kwargs))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
