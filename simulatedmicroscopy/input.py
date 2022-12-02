from __future__ import annotations
from typing import Union
import numpy as np
import warnings


class Coordinates:
    def __init__(self, coordinates: Union[list, np.ndarray]) -> None:
        """Wrapper to hold set of particle coordinates

        Parameters
        ----------
        coordinates : list or np.ndarray
            List of coordinates
        """
        coords = np.array(coordinates)
        if len(coords.shape) != 2:
            raise ValueError("Please give coordinates in (N,3) format in xyz order")

        if coords.shape[1] not in [2, 3]:
            if coords.shape[0] in [2, 3]:
                warnings.warn(
                    "Coords were given in wrong shape, should be an (N,Ndim) array, found (Ndim,N) array, transposing coordinates"
                )
                coords = coords.transpose()
            else:
                raise ValueError("Please give coordinates in (N,3) format in xyz order")

        self.coordinates = coords

    def scale(self, factor: float = 1.0) -> np.ndarray:
        """Update the list of coordinates by multiplication with `factor`

        Parameters
        ----------
        factor : float, optional
            Factor to scale the coordinates by, by default 1.

        Returns
        -------
        np.ndarray
            Numpy array containing the scaled coordinates
        """
        self.coordinates *= factor

        return self.coordinates

    def get_coordinates(self, unit: str = "µm") -> np.ndarray:
        """Get the array containing the particle coordinates

        Parameters
        ----------
        unit : str, optional
            Unit to give the coordinates in. One of [nm, µm], by default "µm"

        Returns
        -------
        np.ndarray
            Array of coordinates (N,3) shape
        """
        scaling_factor = 1.0
        if unit == "nm":
            scaling_factor = 1e3
        return scaling_factor * self.coordinates
