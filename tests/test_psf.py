from simulatedmicroscopy.psf import GaussianPSF
import pytest


def test_can_create_gaussian_psf():
    psf = GaussianPSF([250.0, 250.0, 600.0])

    assert psf.image.sum() > 0.0


def test_can_gaussian_psf_wrongpixelsize():
    with pytest.raises(ValueError):
        GaussianPSF([250.0, 250.0, 600.0], pixel_sizes=[1e-6, 1e-6, 1e-6])
