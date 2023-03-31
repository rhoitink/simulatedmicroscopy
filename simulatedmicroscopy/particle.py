from __future__ import annotations
import numpy as np


class BaseParticle:
    """Base class for all particles with common functions"""

    def __init__(self, pixel_sizes: list[float], dtype: np.dtype = np.uint8) -> None:
        """Initialisation of the particle

        Parameters
        ----------
        pixel_sizes : list[float]
            List of pixel sizes in meters, zyx order
        dtype : np.dtype
            Data type for the response, by default `np.uint8`.
        """
        self.pixel_sizes = np.array(pixel_sizes)
        self.num_dimensions = len(pixel_sizes)
        self.dtype = dtype

    def response(self) -> np.ndarray:
        """Numpy array with the response of the type of particle

        Returns
        -------
        np.ndarray
            Numpy array with intensity values
        """
        return np.array([1, 1, 1], dtype=self.dtype)

    @property
    def shape(self) -> type[tuple]:
        """Shape of the response, i.e. the size of the particle

        Returns
        -------
        type[tuple]
            Shape of the returned particle response array in zyx order
        """
        return self.response().shape


class PointParticle(BaseParticle):
    def __init__(self, pixel_sizes: list[float], *args, **kwargs) -> None:
        super(PointParticle, self).__init__(pixel_sizes, *args, **kwargs)


class Sphere(BaseParticle):
    def __init__(
        self, pixel_sizes: list[float], radius: float, *args, **kwargs
    ) -> None:
        """Sphere with given radius with constant intensity of 1.

        Parameters
        ----------
        pixel_sizes : list[float]
            List of pixel sizes in meters, zyx order
        radius : float
            Radius of the sphere in meters, one float for all directions.
        """
        super(Sphere, self).__init__(pixel_sizes, *args, **kwargs)

        self.radius = radius

        self.radius_px = np.round(self.radius / self.pixel_sizes).astype(int)

        if np.any(self.radius_px < 1):
            raise ValueError(
                "One or more radii are smaller than the pixel size, not possible. Please choose either a smaller pixel size or bigger"
            )

    def response(self):

        z0, y0, x0 = (0, 0, 0)
        zs, ys, xs = np.mgrid[
            -self.radius_px[0] : self.radius_px[0] : 1,
            -self.radius_px[1] : self.radius_px[1] : 1,
            -self.radius_px[2] : self.radius_px[2] : 1,
        ]

        response = np.zeros(shape=zs.shape, dtype=self.dtype)
        response[
            (
                (zs - z0) ** 2 / self.radius_px[0] ** 2
                + (ys - y0) ** 2 / self.radius_px[1] ** 2
                + (xs - x0) ** 2 / self.radius_px[2] ** 2
            )
            < 1.0
        ] = 1

        del zs, ys, xs  # cleanup

        return response
