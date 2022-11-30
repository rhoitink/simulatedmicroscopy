from __future__ import annotations
import numpy as np
from typing import Optional, Union
from pathlib import Path
import h5py
import scipy.signal


class Image:
    DIMENSIONS_ORDER = {"z": 0, "y": 1, "x": 2}

    def __init__(
        self, image: np.ndarray, pixel_sizes: Optional[list[float]] = None
    ) -> None:
        """Initialize an image, used as a wrapper to hold both an N-dimensional
        image and its pixel sizes

        Parameters
        ----------
        image : np.ndarray
            2D or 3D numpy array containing pixel values, order should be (z,)y,x
        pixel_sizes : list[float]
            List/array of pixel sizes in meters, same length as number of image
             dimensions. Order: (z,)y,x. By default 1.0 m/px for all dimensions.
        """
        self.image = np.array(image)

        if pixel_sizes is None:
            # default pixel size of 1 for all dimensions
            self.pixel_sizes = np.array([1.0] * len(self.image.shape))
        else:
            self.pixel_sizes = np.array(pixel_sizes)

        # check if `pixel_sizes` has correct length
        assert (
            self.pixel_sizes.shape[0] == self._number_of_dimensions()
        ), "Not current number of pixel sizes given"

    def _number_of_dimensions(self) -> int:
        """Get the number of dimensions of the image

        Returns
        -------
        int
            Number of dimensions
        """
        return len(self.image.shape)

    def get_pixel_sizes(
        self, dimensions: Optional[list[str]] = list("zyx"), unit: str = "m"
    ) -> list[float]:
        """Get pixel sizes for given dimensions in given unit

        Parameters
        ----------
        dimensions : List of str, optional
            Dimensions to get, one or more of (x, y, z), by default `["z","y","x"]`
        unit : str, optional
            Unit to give the pixel size in. One of [m, cm, mm, µm, um, nm], by default "m".

        Returns
        -------
        list[float]
            List of pixel sizes in given unit, in order of requested dimensions.
        """
        if self._number_of_dimensions() < 3 and "z" in dimensions:
            raise ValueError("Cannot retrieve z dimension for non 3D image")

        dim_indices = [self.DIMENSIONS_ORDER[d] for d in dimensions]

        # fetch pixel sizes in order of requested dimensions
        ps = self.pixel_sizes[dim_indices]

        # convert to requested unit
        if unit == "cm":
            ps *= 1e2
        elif unit == "mm":
            ps *= 1e3
        elif unit == "um" or unit == "µm" or unit == "micron":
            ps *= 1e6
        elif unit == "nm":
            ps *= 1e9

        return ps

    def save_h5file(self, filename: str, description: Optional[str] = None) -> bool:
        """Save image as h5 file (custom format, not compatible with Huygens)

        Parameters
        ----------
        filename : str
            Filename to save the file.
        description : Optional[str], optional
            Description to include in the file, by default None

        Returns
        -------
        bool
            Whether the saving is successfull
        """

        with h5py.File(filename, "w") as f:
            f["Image"] = self.image
            f["Image"].attrs["Description"] = (
                description if description is not None else ""
            )

            f.create_group("Metadata")
            for dim in self.DIMENSIONS_ORDER:
                f[f"Metadata/DimensionScale{dim.upper()}"] = self.get_pixel_sizes(
                    [dim]
                )[0]

        return True

    @classmethod
    def load_h5file(cls, filename: str) -> type[Image]:
        """Load data from h5 file (custom format)

        Parameters
        ----------
        filename : str
            Name of the file to load from

        Returns
        -------
        Image
            Resulting image with correct pixel sizes
        """

        with h5py.File(filename, "r") as f:
            image = f["Image"][()]
            pixel_sizes = [
                float(f[f"Metadata/DimensionScale{dim.upper()}"][()])
                for dim in list("zyx")
            ]

        return cls(image=image, pixel_sizes=pixel_sizes)

    def __eq__(self, other: object) -> bool:
        return (self.image == other.image).all() and (
            self.get_pixel_sizes() == other.get_pixel_sizes()
        ).all()

    def downsample(self, downsample_factor: list[int]) -> type[Image]:
        """Downsample the image, decrease the pixel sizes by given factors.
        Overwrites current image and pixel sizes.

        Parameters
        ----------
        downsample_factor : list[int]
            Factors to downsample each dimension by, in zyx order.

        Returns
        -------
        type[Image]
            Resulting image, original image and pixelsizes are also overwritten
        """
        result = self.image
        for dim in range(self._number_of_dimensions()):
            result = scipy.signal.resample(
                result, self.image.shape[dim] // downsample_factor[dim], axis=dim
            )

        self.image = result.copy()
        del result
        self.pixel_sizes = self.pixel_sizes * np.array(downsample_factor)

        return self

    def convolve(self, other: type[Image]) -> type[Image]:
        """Convolve this image with another image (a PSF). The image is overwritten by the result of the convolution.

        Parameters
        ----------
        other : type[Image]
            The image to convolve this image with

        Returns
        -------
        type[Image]
            The convolved image
        """
        if not (self.pixel_sizes == other.pixel_sizes).all():
            raise ValueError("Cannot convolve images with different pixel sizes")

        self.image = scipy.signal.convolve(self.image, other.image, mode="same")

        return self

    def noisify(self, lam: float = 1.0) -> type[Image]:
        """Add Poisson noise to the image

        Parameters
        ----------
        lam : float, optional
            `lambda` parameter to pass onto the Poisson distribution. Expected number of events occurring in a fixed-time interval, by default 1.0

        Returns
        -------
        type[Image]
            The image with noise added
        """
        rng = np.random.default_rng()
        self.image = self.image * rng.poisson(lam, size=self.image.shape)

        return self


class HuygensImage(Image):
    def __init__(self, filename: Union[str, Path]) -> None:
        """Wrapper for Huygens-generated .h5 images
        Extends the `Image` class.

        Parameters
        ----------
        filename : str
            Name of the Huygens-generated .h5 file
        """

        filepath = Path(filename)
        if not filepath.exists():
            return FileExistsError("Requested file does not exist")

        with h5py.File(filepath, "r") as f:
            image = np.squeeze(f[filepath.stem + "/ImageData/Image"][()])
            pixel_sizes = [
                float(f[filepath.stem + f"/ImageData/DimensionScale{dim}"][()])
                for dim in list("ZYX")
            ]

        super().__init__(image, pixel_sizes)
