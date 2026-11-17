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


@pytest.mark.parametrize('name', ['to_dtype'])
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
            default = getattr(default, '__name__', str(default).removeprefix('torch.'))
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
    for name, kwargs in [('to_dtype', dict(dtype=torch.float32, scale=True)), ('to_dtype', dict(dtype=torch.uint8, scale=False))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.ToDtype(torch.float32, scale=True)]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


def test_dtype_dicts():
    image = F.as_cvcuda_tensor(torch.ones(3,11,13,device='cuda',dtype=torch.uint8))
    expected = torch.full((3,11,13),1/255,device='cuda')
    close(unpack(v2.ToDtype({cvcuda.Tensor:torch.float32},scale=True)(image)), expected, torch.float32)
    assert v2.ToDtype({'others':None})(image) is image
    with pytest.warns(UserWarning, match='no scaling'):
        assert v2.ToDtype({cvcuda.Tensor:None},scale=True)(image) is image
    with pytest.raises(ValueError, match='No dtype'):
        v2.ToDtype({})(image)


@pytest.mark.parametrize('src,dst', [
    (torch.uint8, torch.float16), (torch.uint8, torch.float32),
    (torch.uint8, torch.float64), (torch.uint8, torch.int16),
    (torch.int8, torch.float32), (torch.int16, torch.float32),
    (torch.int32, torch.float32), (torch.int16, torch.uint8),
    (torch.float16, torch.float32),
    (torch.float32, torch.float16), (torch.float32, torch.float64),
    (torch.float64, torch.float32), (torch.float32, torch.uint8),
    (torch.float32, torch.int8), (torch.float32, torch.int16),
])
@pytest.mark.parametrize('scale', [False, True])
def test_dtype_supported_matrix(src, dst, scale):
    if src.is_floating_point:
        values = torch.tensor([0, .0039, .1, .49, .5, .999, 1], device='cuda', dtype=src)
    else:
        maximum = torch.iinfo(src).max
        values = torch.tensor([0, 1, 2, maximum // 4, maximum // 2, maximum - 1, maximum], device='cuda', dtype=src)
        if not scale:
            values = values.remainder(127)
    image = values.repeat(2 * 3 * 5).reshape(2, 3, 5, 7)
    expected = F.to_dtype(image, dtype=dst, scale=scale)
    actual = unpack(F.to_dtype(F.as_cvcuda_tensor(image), dtype=dst, scale=scale))
    assert actual.dtype == dst
    close(actual, expected, torch.uint8 if dst == torch.uint8 else torch.float32)


@pytest.mark.parametrize('scale', [False, True])
def test_dtype_identity(scale):
    image = F.as_cvcuda_tensor(torch.ones(1, 5, 7, device='cuda'))
    assert F.to_dtype(image, dtype=torch.float32, scale=scale) is image


def test_dtype_truncates_instead_of_rounding():
    image = torch.tensor([.9, 1.1, 1.9, 2.5, 126.9], device='cuda').reshape(1, 1, 5)
    actual = unpack(F.to_dtype(F.as_cvcuda_tensor(image), dtype=torch.uint8, scale=False))
    torch.testing.assert_close(actual, image.to(torch.uint8), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.bool, torch.bfloat16])
def test_dtype_unsupported_destination(dtype):
    image = F.as_cvcuda_tensor(torch.ones(1, 5, 7, device='cuda'))
    with pytest.raises(TypeError):
        F.to_dtype(image, dtype=dtype)


_DTYPE_RELEASED_GAPS = [
    pytest.param(torch.int32, torch.int16, [(2**31 - 1) // 4, (2**31 - 1) // 2], True, [8191, 16383], id='scaled-i32-i16-boundary'),
    pytest.param(torch.float16, torch.uint8, [1.], True, [0], id='scaled-f16-u8-at-one'),
    pytest.param(torch.uint16, torch.uint8, [255, 256, 257, 511, 65535], False, [255, 0, 1, 255, 255], id='unscaled-u16-u8-wrap'),
    pytest.param(torch.float64, torch.int32, [.5], True, [1073741823], id='scaled-f64-i32-half'),
]


@pytest.mark.parametrize('src,dst,values,scale,native_values', _DTYPE_RELEASED_GAPS)
def test_dtype_gap_native_reference(src, dst, values, scale, native_values):
    image = torch.tensor(values, device='cuda', dtype=src).reshape(1, 1, -1)
    expected = torch.tensor(native_values, device='cuda', dtype=dst).reshape_as(image)
    torch.testing.assert_close(F.to_dtype(image, dtype=dst, scale=scale), expected, atol=0, rtol=0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='CV-CUDA 0.19 dtype scaling/rounding and saturation differ from TorchVision')
@pytest.mark.parametrize('src,dst,values,scale,native_values', _DTYPE_RELEASED_GAPS)
def test_dtype_released_gap_parity(src, dst, values, scale, native_values):
    _get_kernel(F.to_dtype, cvcuda.Tensor)
    image = torch.tensor(values, device='cuda', dtype=src).reshape(1, 1, -1)
    expected = F.to_dtype(image, dtype=dst, scale=scale)
    actual = unpack(F.to_dtype(F.as_cvcuda_tensor(image), dtype=dst, scale=scale))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_dtype_unsafe_conversion_native_reference():
    image = torch.full((1, 1, 1), .5, device='cuda')
    with pytest.raises(RuntimeError, match='cannot be performed safely'):
        F.to_dtype(image, dtype=torch.int32, scale=True)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='CV-CUDA 0.19 accepts unsafe scaled F32 to I32 conversion')
def test_dtype_unsafe_conversion_error_contract():
    _get_kernel(F.to_dtype, cvcuda.Tensor)
    image = F.as_cvcuda_tensor(torch.full((1, 1, 1), .5, device='cuda'))
    error = None
    try:
        F.to_dtype(image, dtype=torch.int32, scale=True)
    except RuntimeError as exc:
        if 'cannot be performed safely' not in str(exc):
            raise
        error = exc
    assert error is not None, 'Expected TorchVision unsafe-conversion RuntimeError'
