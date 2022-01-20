import pytest
import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.domain import Domain

geometry = Geometry2D()
domain = Domain()


def test_resize_poly():
    a = np.random.uniform(0,100)
    b = np.random.uniform(0,100)
    x_scale = np.random.uniform(0,10)
    y_scale = np.random.uniform(0,10)

    domain = Domain(allowed_area = [(0, 0), (0, b), (a, b), (a, 0)])

    original_poly = domain.bound_poly()
    resized_poly = geometry.resize_poly(original_poly, x_scale = x_scale, y_scale = y_scale)

    resized_square = geometry.get_square(resized_poly)
    original_square = geometry.get_square(original_poly)

    observed_difference = resized_square - original_square
    expected_difference = ((a*x_scale) * (b*y_scale)) - a*b

    assert np.isclose(observed_difference, expected_difference) == True
