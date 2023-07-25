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
        self._response = np.array([[[1]]], dtype=self.dtype)

    def response(self) -> np.ndarray:
        """Numpy array with the response of the type of particle

        Returns
        -------
        np.ndarray
            Numpy array with intensity values
        """
        return self._response

    @property
    def shape(self) -> type[tuple]:
        """Shape of the response, i.e. the size of the particle

        Returns
        -------
        type[tuple]
            Shape of the returned particle response array in zyx order
        """
        return self.response().shape

    @property
    def size(self) -> type[np.ndarray]:
        """Physical size of the response array, in meters for all dimensions

        Returns
        -------
        type[np.ndarray]
            List of physical dimensions in meters in z,y,x order.
        """
        return np.array(self.shape) * self.pixel_sizes


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
                "One or more radii are smaller than the pixel size, not possible. Please choose either a smaller pixel size or bigger radius"
            )

        z0, y0, x0 = (0, 0, 0)
        zs, ys, xs = np.mgrid[
            -self.radius_px[0] : self.radius_px[0] : 1,
            -self.radius_px[1] : self.radius_px[1] : 1,
            -self.radius_px[2] : self.radius_px[2] : 1,
        ]

        self._response = np.zeros(shape=zs.shape, dtype=self.dtype)
        self._response[
            (
                (zs - z0) ** 2 / self.radius_px[0] ** 2
                + (ys - y0) ** 2 / self.radius_px[1] ** 2
                + (xs - x0) ** 2 / self.radius_px[2] ** 2
            )
            < 1.0
        ] = 1

        del zs, ys, xs  # cleanup


class Shell(Sphere):
    def __init__(
        self,
        pixel_sizes: list[float],
        radius: float,
        shell_thickness: float,
        *args,
        **kwargs,
    ) -> None:
        """Shell with given inner radius and shell thickness. Shell has constant intensity of 1., everything else is 0.

        Parameters
        ----------
        pixel_sizes : list[float]
            List of pixel sizes in meters, zyx order
        radius : float
            Inner radius of the shell in meters, one float for all directions.
        shell_thickness : float
            Thickness of the shell in meters, one float for all directions.
        """
        super(Shell, self).__init__(
            pixel_sizes, radius + shell_thickness, *args, **kwargs
        )

        self.shell_thickness = shell_thickness

        self.shell_thickness_px = np.round(
            self.shell_thickness / self.pixel_sizes
        ).astype(int)

        if np.any(self.shell_thickness_px < 1):
            raise ValueError(
                "One or more shell thicknesses are smaller than the pixel size, not possible. Please choose either a smaller pixel size or bigger shell thickness"
            )

        outer_sphere_response = super().response()

        z0, y0, x0 = (0, 0, 0)
        zs, ys, xs = np.mgrid[
            -self.radius_px[0] : self.radius_px[0] : 1,
            -self.radius_px[1] : self.radius_px[1] : 1,
            -self.radius_px[2] : self.radius_px[2] : 1,
        ]

        inner_sphere_response = np.zeros(shape=zs.shape, dtype=self.dtype)
        inner_sphere_response[
            (
                (zs - z0) ** 2 / (self.radius_px[0] - self.shell_thickness_px[0]) ** 2
                + (ys - y0) ** 2 / (self.radius_px[1] - self.shell_thickness_px[1]) ** 2
                + (xs - x0) ** 2 / (self.radius_px[2] - self.shell_thickness_px[2]) ** 2
            )
            < 1.0
        ] = 1

        self._response = outer_sphere_response.copy() - inner_sphere_response.copy()

        del zs, ys, xs, outer_sphere_response, inner_sphere_response  # cleanup


class Spherocylinder(BaseParticle):
    def __init__(
        self, pixel_sizes: list[float], width: float, length: float, *args, **kwargs
    ) -> None:
        """Generation of a spherocylinder with given length and width, oriented along the z axis of the image. Note: the length is excluding the caps. No support for alternative orientation yet.

        Parameters
        ----------
        pixel_sizes : list[float]
            List of pixel sizes in zyx order in meters.
        width : float
            Width of the spherocylinder in meters
        length : float
            Length of the spherocylinder (excluding caps)
        """
        super().__init__(pixel_sizes=pixel_sizes, *args, **kwargs)

        self.radius = width / 2.0
        self.radius_px = np.round(self.radius / self.pixel_sizes).astype(int)

        if np.any(self.radius_px < 1):
            raise ValueError(
                "One or more radii are smaller than the pixel size, not possible. Please choose either a smaller pixel size or bigger radius"
            )

        self.a = length / 2.0
        self.a_px = np.round(self.a / self.pixel_sizes).astype(int)
        if np.any(self.a_px < 1):
            raise ValueError(
                "Length is too short, please increase length or decrease pixel size."
            )

        zs, ys, xs = np.mgrid[
            -self.a_px[0] - self.radius_px[0] : self.a_px[0] + self.radius_px[0] : 1,
            -self.radius_px[1] : self.radius_px[1] : 1,
            -self.radius_px[2] : self.radius_px[2] : 1,
        ]

        self._response = np.zeros(shape=zs.shape, dtype=self.dtype)
        self._response[
            (
                (xs) ** 2 / self.radius_px[2] ** 2
                + (ys) ** 2 / self.radius_px[1] ** 2
                + 0.25
                * (
                    np.abs(zs - self.a_px[0])
                    + np.abs(zs + self.a_px[0])
                    - 2 * self.a_px[0]
                )
                ** 2
                / self.radius_px[0] ** 2
            )
            < 1.0
        ] = 1

        del zs, ys, xs  # cleanup


class Cube(BaseParticle):
    def __init__(self, pixel_sizes: list[float], width: float, *args, **kwargs) -> None:
        """Generate cubic particle

        Parameters
        ----------
        pixel_sizes : list[float]
            List of pixel sizes in meters in zyx order.
        width : float
            Edge length of the cube in meters.
        """
        super().__init__(pixel_sizes, *args, **kwargs)

        self.width = width
        self.width_px = np.round(self.width / self.pixel_sizes).astype(int)

        if np.any(self.width_px < 1):
            raise ValueError(
                "With in one or more dimensions to small, please increase width or decrease pixel size."
            )

        self._response = np.ones(shape=self.width_px, dtype=self.dtype)
