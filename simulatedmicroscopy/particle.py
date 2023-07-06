from __future__ import annotations
import numpy as np

class BaseParticle:
    """Base class for all particles with common functions"""

    def __init__(self) -> None:
        """Initialisation of the particle
        """
        pass

    @property
    def offset(self):
        return 1

    def apply(self, meshgrid: type[np.meshgrid], image: type[np.ndarray],coordinate: type[np.ndarray]) -> None:
        """Apply the response of the particle to the given meshgrid

        Parameters
        ----------
        meshgrid : type[np.meshgrid]
            Numpy meshgrid in all dimensions that is used as coordinate system for the particle image generation
        image : type[np.meshgrid]
            Image that will be updated with the particle's response
        coordinate : type[np.ndarray]
            Coordinate of the particle
        """
        zs,ys,xs = meshgrid
        indices = (np.isclose(zs, coordinate[0]) and np.isclose(ys, coordinate[1]) and np.isclose(xs, coordinate[2]))
        image[indices] += 1


class PointParticle(BaseParticle):
    def __init__(self, *args, **kwargs) -> None:
        super(PointParticle, self).__init__(*args, **kwargs)


class Sphere(BaseParticle):
    def __init__(
        self, radius: float, *args, **kwargs
    ) -> None:
        """Sphere with given radius with constant intensity of 1.

        Parameters
        ----------
        radius : float
            Radius of the sphere in meters, one float for all directions.
        """
        super(Sphere, self).__init__(*args, **kwargs)

        self.radius = radius
    
    def apply(self, meshgrid, image, coordinate):
        indices = np.sum([(meshgrid[i] - coordinate[i])**2 for i in range(3)], axis=0) < self.radius**2

        image[indices] += 1.

    @property
    def offset(self):
        return self.radius
class Shell(BaseParticle):
    pass

# class Shell(Sphere):
#     def __init__(
#         self,
#         pixel_sizes: list[float],
#         radius: float,
#         shell_thickness: float,
#         *args,
#         **kwargs,
#     ) -> None:
#         """Shell with given inner radius and shell thickness. Shell has constant intensity of 1., everything else is 0.

#         Parameters
#         ----------
#         pixel_sizes : list[float]
#             List of pixel sizes in meters, zyx order
#         radius : float
#             Inner radius of the shell in meters, one float for all directions.
#         shell_thickness : float
#             Thickness of the shell in meters, one float for all directions.
#         """
#         super(Shell, self).__init__(
#             pixel_sizes, radius + shell_thickness, *args, **kwargs
#         )

#         self.shell_thickness = shell_thickness

#         self.shell_thickness_px = np.round(
#             self.shell_thickness / self.pixel_sizes
#         ).astype(int)

#         if np.any(self.shell_thickness_px < 1):
#             raise ValueError(
#                 "One or more shell thicknesses are smaller than the pixel size, not possible. Please choose either a smaller pixel size or bigger shell thickness"
#             )

#         outer_sphere_response = super().response()

#         z0, y0, x0 = (0, 0, 0)
#         zs, ys, xs = np.mgrid[
#             -self.radius_px[0] : self.radius_px[0] : 1,
#             -self.radius_px[1] : self.radius_px[1] : 1,
#             -self.radius_px[2] : self.radius_px[2] : 1,
#         ]

#         inner_sphere_response = np.zeros(shape=zs.shape, dtype=self.dtype)
#         inner_sphere_response[
#             (
#                 (zs - z0) ** 2 / (self.radius_px[0] - self.shell_thickness_px[0]) ** 2
#                 + (ys - y0) ** 2 / (self.radius_px[1] - self.shell_thickness_px[1]) ** 2
#                 + (xs - x0) ** 2 / (self.radius_px[2] - self.shell_thickness_px[2]) ** 2
#             )
#             < 1.0
#         ] = 1

#         self._response = outer_sphere_response.copy() - inner_sphere_response.copy()

#         del zs, ys, xs, outer_sphere_response, inner_sphere_response  # cleanup

#     def response(self):

#         return self._response
