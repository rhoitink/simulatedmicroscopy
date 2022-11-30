from simulatedmicroscopy.psf import GaussianPSF
import pytest

def test_can_create_gaussian_psf():
    psf = GaussianPSF([250.,250.,600.])

    assert psf.image.sum() > 0.

def test_can_gaussian_psf_wrongpixelsize():
    with pytest.raises(ValueError):
        GaussianPSF([250.,250.,600.], pixel_sizes=[1e-6, 1e-6, 1e-6])