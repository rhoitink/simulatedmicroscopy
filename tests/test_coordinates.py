from simulatedmicroscopy import Coordinates
import numpy as np
import pytest


def demo_coordinates():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_coordinates():
    coordinates = demo_coordinates()

    cs = Coordinates(coordinates)

    assert (cs.coordinates == coordinates).all()


def test_coordinates_2D():
    coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [2, 0.0], [0.0, 2.0]])

    cs = Coordinates(coordinates)

    assert (cs.coordinates == coordinates).all()


def test_coordinates_wrongshape():
    # instead of given an (N,3) array, give (3,N) array
    coordinates = demo_coordinates().T

    # should atomatically transpose but also give a warning
    with pytest.warns(Warning):
        cs = Coordinates(coordinates)

    assert cs.coordinates.shape[0] == coordinates.shape[1]


def test_coordinates_wrongshape_1d():
    # instead of given an (N,3) array, give 1d array)
    coordinates = demo_coordinates().flat

    # should raise ValueError
    with pytest.raises(ValueError):
        Coordinates(coordinates)


def test_coordinates_wrongshape_5x5():
    # instead of given an (N,3) array, give 5x5 array)
    coordinates = np.zeros(shape=(5, 5))

    # should raise ValueError
    with pytest.raises(ValueError):
        Coordinates(coordinates)


@pytest.mark.parametrize("scaling_factor", [0.3, 1.0, 1.5, 2.0])
def test_scaling(scaling_factor):
    cs = Coordinates(demo_coordinates())

    scaled_coords = cs.scale(scaling_factor)

    assert (cs.coordinates == scaled_coords).all()
    assert (cs.coordinates == demo_coordinates() * scaling_factor).all()


@pytest.mark.parametrize("unit,scaling_factor", [("nm", 1e3), ("µm", 1.0)])
def test_get_coordinates_unit(unit, scaling_factor):
    cs = Coordinates(demo_coordinates())

    converted_coords = cs.get_coordinates(unit=unit)

    assert (converted_coords == demo_coordinates() * scaling_factor).all()
