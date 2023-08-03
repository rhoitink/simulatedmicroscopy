from simulatedmicroscopy import Image, TrackingValidator
import pytest
import numpy as np


def test_image_without_coordinates_raises_error():
    with pytest.raises(ValueError):
        TrackingValidator(
            Image(np.ones((1, 1, 1)), [1e-6, 1e-6, 1e-6]), np.ones((1, 1, 1))
        )


def test_wrong_coordinates_dimension_raises_error():
    with pytest.raises(ValueError):
        im = Image(np.ones((1, 1, 1)), [1e-6, 1e-6, 1e-6])
        im.pixel_coordinates = np.ones((5, 3))
        TrackingValidator(im, np.ones((6, 2)))
