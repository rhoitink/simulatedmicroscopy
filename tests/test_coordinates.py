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


@pytest.mark.parametrize("scaling_factor", [0.3, 1.0, 1.5, 2.0])
def test_scaling(scaling_factor):
    cs = Coordinates(demo_coordinates())

    scaled_coords = cs.scale(scaling_factor)

    assert (cs.coordinates == scaled_coords).all()
    assert (cs.coordinates == demo_coordinates() * scaling_factor).all()
