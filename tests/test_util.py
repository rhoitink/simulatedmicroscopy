import pytest
from simulatedmicroscopy.util import overlap_arrays
import numpy as np


def test_overlap():
    a = np.zeros(shape=(5, 5))
    b = np.zeros(shape=(5, 5))

    c = overlap_arrays(a, b, offset=(5, 5))

    assert c.shape == (10, 10)


def test_overlap_offset0():
    a = np.zeros(shape=(5, 5))
    b = np.zeros(shape=(5, 5))

    c = overlap_arrays(a, b, offset=(0, 0))

    assert c.shape == a.shape


def test_different_shapes():
    a = np.zeros(shape=(5, 5))
    b = np.zeros(shape=(5, 5, 5))

    with pytest.raises(ValueError):
        overlap_arrays(a, b)


def test_wrong_offset():
    a = np.zeros(shape=(5, 5))
    b = np.zeros(shape=(5, 5))

    with pytest.raises(ValueError):
        overlap_arrays(a, b, (0, 0, 0))
