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


@pytest.mark.parametrize('name', ['erase'])
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
    for name, kwargs in [('erase',dict(i=2,j=3,h=4,w=5,v=torch.tensor([.1,.2,.3],device='cuda').view(3,1,1))),('erase',dict(i=2,j=3,h=4,w=5,v=torch.arange(60,device='cuda').reshape(3,4,5).float()))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.RandomErasing(p=1,scale=(.2,.2),ratio=(1.,1.),value=.5)]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('fill_shape', [(1, 1, 1), (3, 1, 1), (3, 4, 5), (2, 3, 4, 5)])
def test_erase_fill_batch_and_inplace(dtype, inplace, fill_shape):
    image = torch.arange(2 * 3 * 11 * 13, device='cuda').reshape(2, 3, 11, 13).remainder(251).to(dtype)
    original = image.clone()
    fill = torch.arange(torch.tensor(fill_shape).prod().item(), device='cuda').reshape(fill_shape).remainder(97).to(dtype)
    inpt = F.as_cvcuda_tensor(image)
    actual = F.erase(inpt, i=2, j=3, h=4, w=5, v=fill, inplace=inplace)
    expected = F.erase(original.clone(), i=2, j=3, h=4, w=5, v=fill)
    torch.testing.assert_close(unpack(actual), expected, atol=0, rtol=0)
    if inplace:
        assert actual is inpt
        torch.testing.assert_close(image, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(image, original, atol=0, rtol=0)


@pytest.mark.parametrize('shape', [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
@pytest.mark.parametrize('inplace', [False, True])
def test_random_erasing_random_value(shape, dtype, inplace):
    torch.manual_seed(23)
    image = torch.rand(shape, device='cuda')
    if dtype == torch.uint8:
        image = (image * 255).to(dtype)
    original = image.clone()
    transform = v2.RandomErasing(p=1, scale=(.2, .4), ratio=(.8, 1.2), value='random', inplace=inplace)
    torch.manual_seed(17)
    expected = transform(original.clone())
    inpt = F.as_cvcuda_tensor(image)
    torch.manual_seed(17)
    actual = transform(inpt)
    torch.testing.assert_close(unpack(actual), expected, atol=0, rtol=0)
    assert not torch.equal(expected, original)
    if inplace:
        assert actual is inpt
        torch.testing.assert_close(image, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(image, original, atol=0, rtol=0)
