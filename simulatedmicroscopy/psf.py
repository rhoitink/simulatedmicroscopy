from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
from .images import HuygensImage, Image
import numpy as np
import scipy.stats


class HuygensPSF(HuygensImage):
    def __init__(self, filename: Union[str, Path]) -> None:
        """Generate PSF from Huygens .h5 file

        Parameters
        ----------
        filename : Union[str, Path]
            Filename/path to the .h5 file
        """
        super().__init__(filename)


class GaussianPSF(Image):
    def __init__(
        self,
        sigmas: list[float],
        pixel_sizes: Optional[list[float]] = [1e-7, 1e-8, 1e-8],
    ) -> None:
        """Generate a 3D Gaussian with given sigmas to use as PSF

        Parameters
        ----------
        sigmas : list[float]
            List of sigmas (in zyx order) to use for the Gaussian distribution, given in nanometers. Please note that this is given in nanometers, while the pixel size is given in meters.
        pixel_sizes : Optional[list[float]], optional
            List of pixel sizes (in zyx order) in meters, by default [1e-7, 1e-8, 1e-8].
        """
        sigmas_nm = np.array(sigmas)
        pixel_sizes_nm = np.round(np.array(pixel_sizes) * 1e9).astype(int)
        
        if not (pixel_sizes_nm < sigmas_nm).all():
            raise ValueError("Pixel sizes should be smaller than given sigmas")
        
        # image size in nm, 4 sigma on all sides of the Gaussian
        image_size_nm = 4 * 2 * sigmas_nm

        # generate slice for the mgrid
        box_slice = [
            slice(-b // 2, b // 2, s) for b, s in zip(image_size_nm, pixel_sizes_nm)
        ]

        z, y, x = np.mgrid[box_slice]
        zyx = np.column_stack([z.flat, y.flat, x.flat])

        # center around zero
        mu = [0.0] * 3

        # generate Gaussian with given mu and sigmas
        gauss_psf = scipy.stats.multivariate_normal.pdf(
            zyx, mu, np.diag(sigmas_nm**2)
        )

        # reshape to 3D image
        gauss_psf = gauss_psf.reshape(z.shape)

        # pass onto Image class with generated image
        super().__init__(gauss_psf, pixel_sizes)
