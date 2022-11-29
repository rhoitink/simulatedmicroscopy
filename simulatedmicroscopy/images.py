import numpy as np
from typing import Optional, Union
from pathlib import Path

class Image:
    DIMENSIONS_ORDER = {'z': 0, 'y': 1, 'x': 2}

    def __init__(self, image : np.ndarray, pixel_sizes : Optional[list[float]] = None) -> None:
        """Initialize an image, used as a wrapper to hold both an N-dimensional
        image and its pixel sizes

        Parameters
        ----------
        image : np.ndarray
            2D or 3D numpy array containing pixel values, order should be (z,)y,x
        pixel_sizes : list[float]
            List/array of pixel sizes in meters, same length as number of image dimensions. Order: (z,)y,x. By default 1.0 m/px for all dimensions.
        """
        self.image = np.array(image)

        if type(pixel_sizes) is type(None):
            # default pixel size of 1 for all dimensions
            self.pixel_sizes = np.array([1.]*len(self.image.shape))
        else:
            self.pixel_sizes = np.array(pixel_sizes)
        
        # check if `pixel_sizes` has correct length
        assert self.pixel_sizes.shape[0] == self._number_of_dimensions(), "Not current number of pixel sizes given"

    def _number_of_dimensions(self) -> int:
        """Get the number of dimensions of the image

        Returns
        -------
        int
            Number of dimensions
        """
        return len(self.image.shape)

    def get_pixel_sizes(self, dimensions : Optional[list[str]] = list("zyx"), unit : str = "m") -> list[float]:
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

    def save_h5file(self, filename : str, description : Optional[str] = None) -> bool:
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
        import h5py

        with h5py.File(filename, 'w') as f:
            f["Image"] = self.image
            f["Image"].attrs['Description'] = description if description is not None else ""
            
            f.create_group("Metadata")
            for dim in self.DIMENSIONS_ORDER:
                f[f"Metadata/DimensionScale{dim.upper()}"] = self.get_pixel_sizes([dim])[0]

        return True

    @classmethod
    def load_h5file(cls, filename : str):
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
        import h5py

        with h5py.File(filename, 'r') as f:
            image = f["Image"][()]
            pixel_sizes = [float(f[f"Metadata/DimensionScale{dim.upper()}"][()]) for dim in list("zyx")]

        return cls(image = image, pixel_sizes = pixel_sizes)

    def __eq__(self, other: object) -> bool:
        return (self.image == other.image).all() and (self.get_pixel_sizes() == other.get_pixel_sizes()).all()


class HuygensImage(Image):

    def __init__(self, filename : Union[str, Path]) -> None:
        """Wrapper for Huygens-generated .h5 images
        Extends the `Image` class.

        Parameters
        ----------
        filename : str
            Name of the Huygens-generated .h5 file
        """
        import h5py

        filepath = Path(filename)
        if not filepath.exists():
            return FileExistsError("Requested file does not exist")

        with h5py.File(filepath, "r") as f:
            image = np.squeeze(f[filepath.stem + "/ImageData/Image"][()])
            pixel_sizes = [float(f[filepath.stem + f"/ImageData/DimensionScale{dim}"][()]) for dim in list("ZYX")]
            
        super().__init__(image, pixel_sizes)