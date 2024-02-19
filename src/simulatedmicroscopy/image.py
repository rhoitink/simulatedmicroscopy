from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import scipy.signal
import skimage.measure

from .input import Coordinates
from .particle import BaseParticle


class Image:
    DIMENSIONS_ORDER = {"z": 0, "y": 1, "x": 2}

    pixel_coordinates = None
    """Pixel coordinates (z,y,x) where particles are positioned"""

    is_convolved = False
    """Whether the image has undergone convolution"""

    is_downsampled = False
    """Whether the image has undergone downsampling"""

    has_shot_noise = False
    """Whether the image has undergone shot noise addition"""

    has_read_noise = False
    """Whether the image has undergone read noise addition"""

    def __init__(
        self,
        image: np.ndarray,
        pixel_sizes: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
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
        metadata : dict
            Place to store custom metadata about an image, will also be stored in .h5 file.
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

        if metadata is None:
            self.metadata = {}
        elif not isinstance(metadata, dict):
            raise ValueError("Metadata should be a dict")
        else:
            self.metadata = metadata

    def _number_of_dimensions(self) -> int:
        """Get the number of dimensions of the image

        Returns
        -------
        int
            Number of dimensions
        """
        return len(self.image.shape)

    def get_pixel_sizes(
        self, dimensions: Optional[list[str]] = None, unit: str = "m"
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
        if dimensions is None:
            dimensions = ["zyx"]

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

    def save_h5file(
        self, filename: str, description: Optional[str] = None
    ) -> bool:
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
                f[
                    f"Metadata/DimensionScale{dim.upper()}"
                ] = self.get_pixel_sizes([dim])[0]

            if self.metadata is not None:
                for k, v in self.metadata.items():
                    f["Metadata"].attrs[k] = v

            # store pixel coordinates if available
            if self.pixel_coordinates is not None:
                f["Metadata/PixelCoordinates"] = self.pixel_coordinates

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
            if "PixelCoordinates" in f["Metadata"]:
                pixel_coordinates = f["Metadata/PixelCoordinates"][()]
            else:
                pixel_coordinates = None

            metadata = dict(f["Metadata"].attrs)

        im = cls(image=image, pixel_sizes=pixel_sizes, metadata=metadata)
        if pixel_coordinates is not None:
            im.pixel_coordinates = pixel_coordinates
        return im

    @staticmethod
    def _get_point_image_array(
        coordinates: type[Coordinates], pixel_sizes: list[float]
    ) -> type[tuple]:
        """Internal method supporting the create_point_image to create a point image array

        Parameters
        ----------
        coordinates : type[CoordinateSet]
            Set of coordinates
        pixel_sizes : list[float]
            List of pixel sizes in meters, in zyx order.

        Returns
        -------
        type[tuple]
            Tuple with ((zs,ys,xs), image)
        """
        # convert pixel sizes to micrometers for calculatation
        pixel_sizes_um = np.array(pixel_sizes) * 1e6

        # scale coordinates with pixel size, order of coords is xyz, while pixel size order is zyx
        scaled_coords = (
            coordinates.get_coordinates().T / pixel_sizes_um[::-1, np.newaxis]
        )

        # round to integer to create point at certain pixel
        xs, ys, zs = np.round(scaled_coords).astype(int)

        # limits for the image, size of each dimension, as coordinates cannot be negative
        limits = (
            zs.max() + 1,
            ys.max() + 1,
            xs.max() + 1,
        )

        image = np.zeros(shape=limits)

        for z, y, x in zip(zs, ys, xs):
            # set pixel value to 1 at location of particles
            image[z, y, x] = 1.0

        return ((zs, ys, xs), image)

    @classmethod
    def create_point_image(
        cls,
        coordinates: type[Coordinates],
        pixel_sizes: list[float],
        *args,
        **kwargs,
    ) -> type[Image]:
        """Create point source image in which every point from the set of coordinates is represented by a single white pixel

        Parameters
        ----------
        coordinates : type[CoordinateSet]
            Set of coordinates
        pixel_sizes : list[float]
            List of pixel sizes in meters, in zyx order.

        Returns
        -------
        type[Image]
            Genereated image
        """
        (zs, ys, xs), image = cls._get_point_image_array(
            coordinates, pixel_sizes
        )

        im = cls(image, pixel_sizes, *args, **kwargs)
        im.pixel_coordinates = np.transpose([zs, ys, xs])
        return im

    @classmethod
    def create_particle_image(
        cls,
        coordinates: type[Coordinates],
        particle: type[BaseParticle],
        edge_pixel_margin: int | list[int] = 0,
        *args,
        **kwargs,
    ) -> type[Image]:
        """Create image in which every point from the set of coordinates is represented by a given `particle`

        Parameters
        ----------
        coordinates : type[CoordinateSet]
            Set of coordinates
        particle : list[BaseParticle]
            Particle to use for image, will also use its pixel size for the final image
        edge_pixel_margin : Union[int,list[int]]
            Number of extra empty pixels to apply as a margin around the edges of the image. Either single int for all directions or one int for
            every dimension (in z,y,x order).

        Returns
        -------
        type[Image]
            Genereated image
        """
        # convert pixel sizes to micrometers for calculatation
        pixel_sizes_um = np.array(particle.pixel_sizes) * 1e6
        if isinstance(edge_pixel_margin, (int, float)):
            # not iterable, repeat for all dims
            edge_pixel_margin = np.array(
                [edge_pixel_margin] * len(pixel_sizes_um), dtype=int
            )
        else:
            if len(edge_pixel_margin) != len(pixel_sizes_um):
                raise ValueError(
                    f"Length of edge_pixel_margin not suitable, please make sure you pass {len(pixel_sizes_um)} or 1 integers"
                )
            edge_pixel_margin = np.array(edge_pixel_margin, dtype=int)

        # scale coordinates with pixel size, order of coords is xyz, while pixel size order is zyx
        scaled_coords = (
            coordinates.get_coordinates().T / pixel_sizes_um[::-1, np.newaxis]
        )

        particle_offset = (
            np.array(particle.shape) / 2.0
        )  # offset coordinates by half the size of the box, such that the coordinate points to the middle of the particle

        # round to integer to create point at certain pixel
        xs, ys, zs = (
            np.round(scaled_coords).astype(int)
            + edge_pixel_margin[::-1, np.newaxis]
        )
        # edge_pixel_margin to x,y,z order by reversing

        particle_response = particle.response().copy()
        prs = np.array(particle.shape)

        image_size = np.ceil(
            np.max([zs, ys, xs], axis=1) + prs + edge_pixel_margin
        ).astype(int)

        image = np.zeros(shape=image_size)

        for x, y, z in zip(xs, ys, zs):
            image[
                z : z + prs[0], y : y + prs[1], x : x + prs[2]
            ] += particle_response

        im = cls(image, particle.pixel_sizes, *args, **kwargs)
        im.pixel_coordinates = np.transpose([zs, ys, xs]) + particle_offset.T
        return im

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
        # can only deal with integers, so round and cast to int
        ds_fac = np.round(downsample_factor).astype(int)

        # reduce image size by taking the mean of a AxBxC (defined by downsample_factor) sized block and use that as new pixel value
        self.image = skimage.measure.block_reduce(
            self.image, tuple(ds_fac), np.mean
        )

        # adapt pixel sizes
        self.pixel_sizes = self.pixel_sizes * ds_fac

        # adapt particle coordinates
        if self.pixel_coordinates is not None:
            self.pixel_coordinates = self.pixel_coordinates // ds_fac

        self.is_downsampled = True
        self.metadata["is_downsampled"] = True

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
        if not np.isclose(self.pixel_sizes, other.pixel_sizes).all():
            raise ValueError(
                "Cannot convolve images with different pixel sizes"
            )

        self.image = scipy.signal.convolve(
            self.image, other.image, mode="same"
        )

        self.is_convolved = True
        self.metadata["is_convolved"] = True

        return self

    def noisify(self, lam: float = 1.0) -> type[Image]:
        """Add Poisson noise to the image [DEPRECATED]

        Parameters
        ----------
        lam : float, optional
            `lambda` parameter to pass onto the Poisson distribution. Expected number of events occurring in a fixed-time interval, by default 1.0

        Returns
        -------
        type[Image]
            The image with noise added
        """
        raise DeprecationWarning(
            "This method is deprecated, please use either add_shot_noise or add_read_noise"
        )

    def add_shot_noise(self, SNR: float = 30.0) -> type[Image]:
        """Add shot noise to the image, realised by sampling the intensities of the image from a Poisson distribution.
        At each pixel, the current value (after rescaling) dictates the mean value of the Poisson distribution, yielding
        signal-dependent noise.

        Parameters
        ----------
        SNR : float, optional
            The signal-to-noise ratio of the resulting image, by default 30.0.

        Returns
        -------
        type[Image]
            The image with shot noise
        """
        if self.has_shot_noise:
            warnings.warn(
                "This image already has shot noise, proceeding anyway",
                stacklevel=2,
            )

        # Poisson cannot deal with negative values, if these are present, shift whole image
        if np.any(self.image < 0.0):
            self.image += np.abs(self.image.min())

        scaling_factor = (
            SNR**2 / self.image.mean()
        )  # scaling factor to get a proper Poisson sampling

        self.image = (
            np.random.poisson(scaling_factor * self.image) / scaling_factor
        )
        # division by `scaling_factor` at the end to revert pixel values back to their (roughly) original range

        self.has_shot_noise = True  # set shot noise flag

        return self

    def add_read_noise(
        self, SNR: float = 50.0, background: float = 0.0
    ) -> type[Image]:
        """Add read noise to the image, realised by adding noise sampled from a normal distribution. The standard deviation
        of the distribution is constant across the entire image and determined by the SNR parameter, it is therefore
        signal-independent.

        Parameters
        ----------
        SNR : float, optional
            The signal-to-noise ratio of the resulting image, by default 50.0
        background : float, optional
            Optional offset of the background, this will be the mean of the normal distribution, by default 0.0

        Returns
        -------
        type[Image]
            The image with read noise
        """
        if self.has_read_noise:
            warnings.warn(
                "This image already has read noise, proceeding anyway",
                stacklevel=2,
            )

        peak = np.percentile(
            self.image, 99.9
        )  # get peak intensity at the 99.9 percentile

        # As SNR = peak / sigma; this yields that sigma = peak/SNR
        sigma = peak / SNR

        self.image += np.random.normal(
            loc=background, scale=sigma, size=self.image.shape
        )  # centred around value of `background`

        self.has_read_noise = True  # set read noise flag

        return self

    def get_pixel_coordinates(self) -> np.ndarray:
        """Get list of pixel indices containing particles

        Returns
        -------
        np.ndarray
            Numpy (N,3) array in zyx order with indices of pixels containing particles
        """
        return self.pixel_coordinates

    def get_particle_image(
        self, particle_id: int = 0, box_size: tuple[int] = (5, 5, 5)
    ) -> np.ndarray:
        """Get 3D box around the coordinates of particle with index `particle_id`

        Parameters
        ----------
        particle_id : int, optional
            Index of the particle, by default 0
        box_size : tuple[int], optional
            Size of the surrounding box in pixels, by default (5, 5, 5)

        Returns
        -------
        np.ndarray
            Image containing the particle with shape determined by `box_size`. Box will be clipped to the the limits of the images and can therefore be smaller than `box_size`.
        """
        if self.pixel_coordinates is None:
            raise ValueError(
                "Not a image genereted from particle, cannot extract single particles"
            )

        if int(particle_id) < 0 or int(particle_id) >= len(
            self.pixel_coordinates
        ):
            raise IndexError("Invalid particle index")

        coords = self.pixel_coordinates[int(particle_id)].astype(int)
        slices = []
        for dim in range(len(self.image.shape)):
            # check whether the indices will not go outside the image, else clip them back to the image boundaries (will result in smaller box)
            left = max(0, coords[dim] - box_size[dim] // 2)
            right = min(self.image.shape[dim], left + box_size[dim])
            slices.append(slice(left, right, 1))

        return self.image[tuple(slices)]


class HuygensImage(Image):
    def __init__(self, filename: str | Path) -> None:
        """Wrapper for Huygens-generated .h5 images
        Extends the `Image` class.

        Parameters
        ----------
        filename : str
            Name of the Huygens-generated .h5 file
        """

        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError("Requested file does not exist")

        with h5py.File(filepath, "r") as f:
            image = np.squeeze(f[filepath.stem + "/ImageData/Image"][()])
            pixel_sizes = [
                float(f[filepath.stem + f"/ImageData/DimensionScale{dim}"][()])
                for dim in list("ZYX")
            ]

        super().__init__(image, pixel_sizes)
