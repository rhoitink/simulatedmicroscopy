from simulatedmicroscopy.particle import Sphere, Shell
import pytest
import numpy as np


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
