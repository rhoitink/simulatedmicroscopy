import numpy as np

class CoordinateSet:

    def __init__(self, coordinates : list or np.ndarray) -> None:
        """Wrapper to hold set of particle coordinates

        Parameters
        ----------
        coordinates : list or np.ndarray
            List of coordinates
        """
        self.coordinates = np.array(coordinates)

    def scale(self, factor : float = 1.) -> np.ndarray:
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
