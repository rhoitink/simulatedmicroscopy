import numpy as np


def overlap_arrays(
    a: np.ndarray, b: np.ndarray, offset: tuple = (0, 0, 0)
) -> np.ndarray:
    """
    Make array that sums arrays a and b with given offset

    Parameters
    ----------
    a : np.ndarray
        Array 1, this serves as base and starts at index 0
    b : np.ndarray
        Array to be added, needs same number of axes as a.
    offset : list of ints, optional
        Offset position, that determines where addition of b to a starts.
        Provide a list with an offset index for every axis in the arrays.
        Only positive numbers allowed
        The default is (0,0,0).

    Raises
    ------
    ValueError
        If both errors do not have the same number of axes.

    Returns
    -------
    type[np.ndarray]
        Result of the addition.

    """
    s_a = a.shape
    s_b = b.shape
    if len(s_a) != len(s_b):
        raise ValueError("Arrays do not have the same number of axes, cannot overlap")
    if len(s_a) > 1:
        if len(offset) != len(s_a):
            raise ValueError(
                "Offset not included for every axis, "
                "make sure offset is same length as number of dimensions"
            )
    s_c = [max(a.shape[i], offset[i] + b.shape[i]) for i in range(len(s_a))]

    c = np.zeros(shape=s_c, dtype=a.dtype)

    # add array a with index 0 (in all dims)
    # for indexing, tuple is needed
    slices_a = tuple([slice(0, s_a[dim]) for dim in range(len(c.shape))])

    # add array b, starting at offset index
    slices_b = tuple(
        [slice(offset[dim], offset[dim] + s_b[dim]) for dim in range(len(c.shape))]
    )

    c[slices_a] += a.copy()
    c[slices_b] += b.copy()

    return c
