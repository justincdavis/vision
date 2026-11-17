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


@pytest.mark.parametrize('name', ['crop', 'center_crop', 'resized_crop', 'five_crop', 'ten_crop'])
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
    for name, kwargs in [('crop', dict(top=-2,left=-3,height=17,width=19)), ('center_crop',dict(output_size=[17,19])), ('resized_crop',dict(top=1,left=2,height=7,width=9,size=[5,6])), ('five_crop',dict(size=[7,9])), ('ten_crop',dict(size=[7,9])), ('ten_crop',dict(size=[7,9],vertical_flip=True))]:
        expected = getattr(F,name)(image.clone(), **kwargs)
        actual = unpack(getattr(F,name)(F.as_cvcuda_tensor(image.clone()), **kwargs))
        close(actual, expected, dtype)
    for transform in [v2.CenterCrop((17,19)),v2.FiveCrop((7,9)),v2.TenCrop((7,9)),v2.RandomCrop((7,9)),v2.RandomResizedCrop((7,9),scale=(.4,.9),ratio=(.8,1.2))]:
        torch.manual_seed(17)
        expected = transform(image.clone())
        torch.manual_seed(17)
        actual = transform(F.as_cvcuda_tensor(image.clone()))
        assert isinstance(actual, (cvcuda.Tensor, tuple))
        close(unpack(actual), expected, dtype)


@pytest.mark.parametrize('size', [[12, 13], [11, 14]])
def test_five_crop_oversize_released_error(size):
    image = torch.ones(3, 11, 13, device='cuda')
    with pytest.raises(ValueError, match='Requested crop size'):
        F.five_crop(image, size=size)
    # CV-CUDA reports the backend ROI error rather than TorchVision's ValueError.
    with pytest.raises(RuntimeError, match='Invalid Roi range'):
        F.five_crop(F.as_cvcuda_tensor(image), size=size)


@pytest.mark.parametrize('shape', [(3,11,13), (2,3,11,13)])
@pytest.mark.parametrize('size', [8, [8], [8,10], [12,16]])
def test_center_crop_size_parsing_and_odd_offsets(shape, size):
    image = torch.arange(torch.tensor(shape).prod().item(), device='cuda').reshape(shape).float()
    expected = F.center_crop(image, output_size=size)
    actual = unpack(F.center_crop(F.as_cvcuda_tensor(image), output_size=size))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(unpack(v2.CenterCrop(size)(F.as_cvcuda_tensor(image))), expected, atol=0, rtol=0)
