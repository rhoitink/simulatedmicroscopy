import numpy as np
import pytest
from simulatedmicroscopy import HuygensImage, Image, HuygensPSF


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


def test_huygens_notexisting(tmp_path):
    with pytest.raises(FileNotFoundError):
        HuygensImage(tmp_path / "thisfiledoesnotexist.h5")

def test_huygens_loading(tmp_path):
    import h5py
    im = create_demo_image()
    pixel_sizes = im.get_pixel_sizes()
    filename = tmp_path / "testfile.hdf5"
    with h5py.File(filename, "w") as f:
        root = f.create_group("testfile")
        root.create_dataset("ImageData/Image", data=im.image)
        [root.create_dataset(f"ImageData/DimensionScale{dim}", data=pixel_sizes[i]) for i,dim in enumerate(list("ZYX"))]
    
    
    im_loaded = im.load_h5file(filename)
    im_loaded_huygens_im = HuygensImage(filename)
    im_loaded_huygens_psf = HuygensPSF(filename)


    assert im_loaded == im
    assert im_loaded_huygens_im == im
    assert im_loaded_huygens_psf == im

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
    assert (
        image.get_pixel_sizes(unit=unit) == pixel_sizes * conversionfactor
    ).all()


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
        downsampled.image.shape
        == np.array(im_array.shape) // downsample_factor
    ).all()

    # check if the pixel_size of the image has increased by `downsample_factor`
    assert (im.pixel_sizes == np.array([downsample_factor] * 3)).all()

    # check if the original image was also updated
    assert downsampled == im

    # should be set to True
    assert im.is_downsampled


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

    # should be set to True
    assert im.is_convolved


def test_convolution_wrongpixelsize():
    im = create_demo_image()
    psf = Image(np.ones(shape=(1, 1, 1)), pixel_sizes=(10.0, 10.0, 10.0))

    with pytest.raises(ValueError):
        im.convolve(psf)


def test_noise_deprecated():
    with pytest.raises(DeprecationWarning):
        im = create_demo_image()
        im.noisify()


def test_shot_noise():
    im = create_demo_image()

    # check if returns correct image
    assert im.add_shot_noise(SNR=10.0) != create_demo_image()

    # check if original image was also changed
    assert im != create_demo_image()

    # should be set to True
    assert im.has_shot_noise


def test_read_noise():
    im = create_demo_image()

    # check if returns correct image
    assert im.add_read_noise(SNR=10.0, background=1e-3) != create_demo_image()

    # check if original image was also changed
    assert im != create_demo_image()

    # should be set to True
    assert im.has_read_noise


def test_noise_apply_both():
    im = create_demo_image()

    assert im.add_read_noise().add_shot_noise() != create_demo_image()

    assert im.has_read_noise

    assert im.has_shot_noise


def test_point_image(tmp_path):
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
    assert im.pixel_coordinates is not None

    # check if we can restore the pixel coordinates from the h5 file
    fp = tmp_path / "test_pointimage.h5"
    im.save_h5file(fp)

    im2 = im.load_h5file(fp)

    assert im2.pixel_coordinates is not None
    assert im == im2


@pytest.mark.parametrize(
    "downsample_factor",
    [
        1,
        2,
        10,
    ],
)
def test_pixel_coordinates_after_downscale(downsample_factor):
    from simulatedmicroscopy import Coordinates

    cs = Coordinates(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    im = Image.create_point_image(cs, [100e-9, 10e-9, 10e-9])

    pixel_coords_before = im.get_pixel_coordinates().copy()

    im.downsample([downsample_factor] * 3)

    assert (
        pixel_coords_before == im.get_pixel_coordinates() * downsample_factor
    ).all()


def test_pixel_coordinates_after_downscale_onlyz():
    downsample_factor = 2
    from simulatedmicroscopy import Coordinates

    cs = Coordinates(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    im = Image.create_point_image(cs, [100e-9, 10e-9, 10e-9])

    pixel_coords_before = im.get_pixel_coordinates().copy()

    im.downsample([downsample_factor, 1, 1])

    assert (
        pixel_coords_before
        == im.get_pixel_coordinates() * np.transpose([downsample_factor, 1, 1])
    ).all()


def test_image_metadata_wrongtype():
    with pytest.raises(ValueError):
        Image(
            np.zeros(shape=(5, 5, 5)), [1e-6, 1e-6, 1e-6], metadata=[1, 2, 3]
        )


def test_image_metadata_is_set():
    meta = {"key1": "val1", "key2": 2, "key3": 3.0}
    im = Image(np.zeros(shape=(5, 5, 5)), [1e-6, 1e-6, 1e-6], metadata=meta)

    assert im.metadata == meta


def test_image_metadata_can_save_and_retrieve(tmp_path):
    meta = {"key1": "val1", "key2": 2, "key3": 3.0}

    im = Image(np.zeros(shape=(5, 5, 5)), [1e-6, 1e-6, 1e-6], metadata=meta)
    im.save_h5file(tmp_path / "test_metadata.h5")

    im2 = Image.load_h5file(tmp_path / "test_metadata.h5")

    assert im2.metadata == meta


def test_image_metadata_works_on_classmethods():
    from simulatedmicroscopy import Coordinates, Sphere

    meta = {"key1": "val1", "key2": 2, "key3": 3.0}

    cs = Coordinates(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    point_im = Image.create_point_image(cs, [1e-6, 1e-6, 1e-6], metadata=meta)

    particle_im = Image.create_particle_image(
        cs, Sphere([1e-6, 1e-6, 1e-6], 1e-6), metadata=meta
    )

    assert point_im.metadata == meta
    assert particle_im.metadata == meta


def test_particle_image_desired_shape():
    from simulatedmicroscopy import Coordinates, Sphere

    cs = Coordinates(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [2.5, 2.5, 2.5],
            [0.0, 5.0, 0.0],
        ]
    )

    particle_im = Image.create_particle_image(
        cs, Sphere([1e-6, 1e-6, 1e-6], 1e-6)
    )

    box_shape = (2, 2, 2)

    assert particle_im.get_particle_image(2, box_shape).shape == box_shape


def test_particle_image_invalid_index():
    from simulatedmicroscopy import Coordinates, Sphere

    cs = Coordinates(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    pi = Image.create_particle_image(cs, Sphere([1e-6, 1e-6, 1e-6], 1e-6))

    with pytest.raises(IndexError):
        pi.get_particle_image(-1)
        pi.get_particle_image(len(cs.coordinates))
        pi.get_particle_image(len(cs.coordinates) + 1)


def test_particle_image_invalid_image():
    with pytest.raises(ValueError):
        Image(np.zeros((3, 3, 3))).get_particle_image()
