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


@pytest.mark.parametrize('name', ['elastic'])
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
    for name, kwargs in [('elastic',dict(displacement=torch.zeros(1,11,13,2,device='cuda'),fill=7)),('elastic',dict(displacement=torch.full((1,11,13,2),2.,device='cuda'),fill=7))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.ElasticTransform(alpha=2.,sigma=1.,fill=7)]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('shape', [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize('dtype,interpolation', [
    (torch.uint8, InterpolationMode.NEAREST),
    (torch.uint8, InterpolationMode.BILINEAR),
    (torch.float32, InterpolationMode.NEAREST),
    pytest.param(torch.float32, InterpolationMode.BILINEAR, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason='Released elastic bilinear F32 range [0,250] exceeds 2e-5 (up to 3.36e-4)'
    )),
])
@pytest.mark.parametrize('pixels', [0., 1., .25])
def test_elastic_known_displacement(shape, dtype, interpolation, pixels):
    image = torch.arange(torch.tensor(shape).prod().item(), device='cuda').reshape(shape).remainder(251).to(dtype)
    displacement = torch.zeros(1, 11, 13, 2, device='cuda')
    displacement[..., 0] = 2 * pixels / 13
    kwargs = dict(displacement=displacement, interpolation=interpolation, fill=7)
    expected = F.elastic(image, **kwargs)
    actual = unpack(F.elastic(F.as_cvcuda_tensor(image), **kwargs))
    direct = unpack(cvcuda.tv.elastic(F.as_cvcuda_tensor(image), **kwargs))
    torch.testing.assert_close(actual, direct, atol=0, rtol=0)
    close(actual, expected, dtype)
    if interpolation == InterpolationMode.NEAREST and pixels in (0., 1.):
        known = image.clone()
        if pixels == 1.:
            known[..., :-1] = image[..., 1:]
            known[..., -1] = 7
        torch.testing.assert_close(actual, known, atol=0, rtol=0)


@pytest.mark.parametrize('shape', [(11, 13, 2), (2, 11, 13, 2), (1, 10, 13, 2), (1, 11, 13, 3)])
def test_elastic_displacement_shape_errors(shape):
    image = torch.ones(3, 11, 13, device='cuda')
    displacement = torch.zeros(shape, device='cuda')
    for inpt in [image, F.as_cvcuda_tensor(image)]:
        with pytest.raises(ValueError, match='Argument displacement shape should be'):
            F.elastic(inpt, displacement)


@pytest.mark.parametrize('displacement', [None, 1, [[0., 0.]]])
def test_elastic_native_displacement_type_errors(displacement):
    image = torch.ones(3, 11, 13, device='cuda')
    with pytest.raises(TypeError):
        F.elastic(image, displacement)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Released elastic leaks AttributeError for non-tensor displacement instead of TypeError')
@pytest.mark.parametrize('displacement', [None, 1, [[0., 0.]]])
def test_elastic_displacement_type_errors(displacement):
    image = torch.ones(3, 11, 13, device='cuda')
    with pytest.raises(TypeError):
        try:
            F.elastic(F.as_cvcuda_tensor(image), displacement)
        except AttributeError as error:
            expected = f"'{type(displacement).__name__}' object has no attribute 'shape'"
            if str(error) != expected:
                raise
            raise AssertionError('Known released displacement type-validation gap') from error


@pytest.mark.parametrize('shape', [(3, 11, 13), (2, 3, 11, 13)])
@pytest.mark.parametrize('interpolation', [InterpolationMode.NEAREST, InterpolationMode.BILINEAR])
@pytest.mark.parametrize('pixels', [0., 1., .25])
def test_elastic_unit_interval_displacement(shape, interpolation, pixels):
    torch.manual_seed(23)
    image = torch.rand(shape, device='cuda')
    displacement = torch.zeros(1, 11, 13, 2, device='cuda')
    displacement[..., 0] = 2 * pixels / 13
    kwargs = dict(displacement=displacement, interpolation=interpolation, fill=.2)
    close(unpack(F.elastic(F.as_cvcuda_tensor(image), **kwargs)), F.elastic(image, **kwargs), torch.float32)
