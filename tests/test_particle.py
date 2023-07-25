from simulatedmicroscopy.particle import Sphere, Shell, PointParticle, Spherocylinder
from simulatedmicroscopy.image import Image
from simulatedmicroscopy.input import Coordinates
import pytest
import numpy as np


def test_can_create_particle():
    p = PointParticle([1e-6, 1e-6, 1e-6])

    assert p.response().shape == (1, 1, 1)


@pytest.mark.parametrize(
    "radius",
    [
        2e-6,
        1e-6,
        1e-5,
    ],
)
def test_can_create_sphere(radius):
    sphere = Sphere([1e-6, 1e-6, 1e-6], radius)
    assert sphere.response().sum() > 0.0


def test_sphere_response():
    sphere = Sphere([1e-6, 1e-6, 1e-6], 2e-6)
    assert np.sum(sphere.response()) > 0.0


def test_particle_has_shape():
    sphere = Sphere([1e-6, 1e-6, 1e-6], 2e-6)

    assert isinstance(sphere.shape, tuple)


def test_particle_has_size():
    sphere = Sphere([1e-6, 1e-6, 1e-6], 2e-6)

    assert isinstance(sphere.size, np.ndarray)


def test_sphere_wrong_radius_format():
    with pytest.raises(ValueError):
        Sphere([1e-6, 1e-6, 1e-6], [5e-6, 5e-6])


def test_sphere_too_small_radius():
    with pytest.raises(ValueError):
        Sphere([1e-6, 1e-8, 1e-8], 1e-8)


def test_can_create_shell():
    shell = Shell([1e-6, 1e-6, 1e-6], 2e-6, 1e-6)
    assert np.sum(shell.response()) > 0.0


def test_too_thin_shell():
    with pytest.raises(ValueError):
        Shell([1e-6, 1e-6, 1e-6], 2e-6, 1e-7)


def test_can_create_spherocylinder():
    sc = Spherocylinder([1e-6, 1e-8, 1e-8], 2e-6, 5e-6)
    assert np.sum(sc.response()) > 0.0


def test_spherocylinder_longer_than_wide():
    sc = Spherocylinder([1e-6, 1e-6, 1e-6], 2e-6, 5e-6)
    assert sc.shape[0] > sc.shape[1] and sc.shape[0] > sc.shape[2]


def test_can_spherocylinder_to_narrow():
    with pytest.raises(ValueError):
        Spherocylinder([1e-6, 1e-8, 1e-8], 1e-6, 5e-6)


def test_can_spherocylinder_to_short():
    with pytest.raises(ValueError):
        Spherocylinder([1e-6, 1e-8, 1e-8], 2e-6, 1e-6)


def test_can_create_image():
    particle = Sphere([1e-6, 1e-6, 1e-6], 2e-6)
    coords = Coordinates([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    im = Image.create_particle_image(coords, particle)
    assert im.image.sum() > 0.0


def test_particle_image_values():
    particle = Sphere([1e-6, 1e-6, 1e-6], 2e-6)
    coords = Coordinates([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    im = Image.create_particle_image(coords, particle)
    # summed pixel intensity should be twice the single particle intensity
    # for two overlapping identical particles
    assert im.image.sum() == 2.0 * particle.response().sum()


def test_particle_image_size():
    diameter_um = 2.0
    particle = Sphere([1e-6, 1e-6, 1e-6], 1e-6 * diameter_um / 2.0)
    coords = Coordinates([[0.0, 0.0, 0.0], [0.0, diameter_um, 0.0]])
    im = Image.create_particle_image(coords, particle)
    # image size should be twice the particle size when placed 1 diameter apart
    assert np.prod(im.image.shape) == 2.0 * np.prod(particle.shape)


def test_particle_image_edge_pixel_margin():
    diameter_um = 2.0
    edge_pixel_margin = 10
    particle = Sphere([1e-6, 1e-6, 1e-6], 1e-6 * diameter_um / 2.0)
    coords = Coordinates([[0.0, 0.0, 0.0], [0.0, diameter_um, 0.0]])
    im = Image.create_particle_image(coords, particle)
    im2 = Image.create_particle_image(coords, particle, edge_pixel_margin)

    # image shape should be padded on all sides by edge_pixel_margin
    assert np.all(
        np.array(im2.image.shape) == 2 * edge_pixel_margin + np.array(im.image.shape)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        int,
        np.float16,
    ],
)
def test_can_change_dtype(dtype):
    testarr = np.empty(shape=1, dtype=dtype)
    assert (
        Sphere([1e-6, 1e-6, 1e-6], 2e-6, dtype=dtype).response().dtype == testarr.dtype
    )
