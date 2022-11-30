from simulatedmicroscopy import Image
import numpy as np
import pytest


def create_demo_image():
    return Image(np.ones(shape=(10, 10, 10)), pixel_sizes=(1e-7, 1e-8, 1e-8))


def test_3D_image():
    im_array = np.zeros(shape=(5, 5, 5))
    image = Image(im_array)

    assert (image.image == im_array).all()


def test_2D_image():
    im_array = np.zeros(shape=(5, 5))
    image = Image(im_array)

    assert (image.image == im_array).all()


def test_2D_image_getzsize():
    im_array = np.zeros(shape=(5, 5))
    image = Image(im_array)
    with pytest.raises(ValueError):
        image.get_pixel_sizes(dimensions=["z"])


def test_pixel_sizes():
    pixel_sizes = np.array([1e-7, 1e-8, 1e-9])
    image = Image(np.zeros(shape=(5, 5, 5)), pixel_sizes)

    assert (image.pixel_sizes == pixel_sizes).all()
    assert (image.get_pixel_sizes() == pixel_sizes).all()


@pytest.mark.parametrize(
    "unit,conversionfactor",
    [
        ("m", 1.0),
        ("cm", 1e2),
        ("mm", 1e3),
        ("um", 1e6),
        ("µm", 1e6),
        ("micron", 1e6),
        ("nm", 1e9),
    ],
)
def test_pixel_size_conversion(unit, conversionfactor):
    pixel_sizes = np.array([1e-9, 1e-8, 1e-7])
    image = Image(np.zeros(shape=(5, 5, 5)), pixel_sizes)
    assert (image.get_pixel_sizes(unit=unit) == pixel_sizes * conversionfactor).all()


def test_image_equality_image():
    im_array1 = np.zeros(shape=(5, 5, 5))
    image1 = Image(im_array1)

    im_array2 = np.zeros(shape=(5, 5, 5))
    im_array2[0] = 1.0
    image2 = Image(im_array2)

    image3 = Image(im_array1)

    assert image1 != image2
    assert image1 == image3


def test_image_equality_pixel_size():
    im_array = np.zeros(shape=(5, 5, 5))
    image = Image(im_array)

    assert (image.image == im_array).all()


def test_h5file_saving(tmp_path):
    filepath = tmp_path / "test_outputfile.h5"
    im = create_demo_image()

    im.save_h5file(filepath)

    assert filepath.exists()


def test_h5file_loading(tmp_path):
    test_h5file_saving(tmp_path)
    filepath = tmp_path / "test_outputfile.h5"
    im = Image.load_h5file(filepath)

    assert im == create_demo_image()


@pytest.mark.parametrize(
    "downsample_factor",
    [
        1,
        2,
        5,
    ],
)
def test_downsample(downsample_factor):
    im_array = np.zeros(shape=(100, 200, 500))
    im = Image(im_array)

    downsampled = im.downsample([downsample_factor] * 3)

    # check if the shape of the image has decreased by `downsample_factor`
    assert (
        downsampled.image.shape == np.array(im_array.shape) // downsample_factor
    ).all()

    # check if the pixel_size of the image has increased by `downsample_factor`
    assert (im.pixel_sizes == np.array([downsample_factor] * 3)).all()

    # check if the original image was also updated
    assert downsampled == im


@pytest.mark.parametrize(
    "multiplication_factor",
    [
        0.5,
        1.0,
        2.0,
    ],
)
def test_convolution(multiplication_factor):
    im = Image(np.ones(shape=(10, 10, 10)))
    psf = Image(np.ones(shape=(1, 1, 1)) * multiplication_factor)

    convolved = im.convolve(psf)

    assert im == convolved


def test_convolution_wrongpixelsize():
    im = create_demo_image()
    psf = Image(np.ones(shape=(1, 1, 1)), pixel_sizes=(10.0, 10.0, 10.0))

    with pytest.raises(ValueError):
        im.convolve(psf)


def test_noise():
    im = create_demo_image()

    # check if returns correct image
    assert im.noisify() != create_demo_image()

    # check if original image was also changed
    assert im != create_demo_image()


def test_point_image():
    from simulatedmicroscopy import Coordinates

    cs = Coordinates(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    im = Image.create_point_image(cs, [100e-9, 10e-9, 10e-9])

    assert im.image.sum() > 0.0
