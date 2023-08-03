from .image import Image
import numpy as np


class TrackingValidator:
    def __init__(self, image: Image, coordinates: np.ndarray) -> None:
        """Class to compare the accuracy of particle tracking to the ground truth particle coordinates

        Parameters
        ----------
        image : Image
            Image with particle coordinates embedded
        coordinates : np.ndarray
            (N,NDIM) array containing the pixel locations for each tracked particle
        """

        self.image = image
        self.coordinates = np.array(coordinates)

        if self.image.get_pixel_coordinates() is None:
            raise ValueError("Image object does not contain particle coordinates")

        if self.image.get_pixel_coordinates().shape[-1] != self.coordinates.shape[-1]:
            raise ValueError(
                f"Entered coordinates do not have the right dimensions, should be: (N, {self.image.get_pixel_coordinates().shape[-1]}) for this image"
            )
