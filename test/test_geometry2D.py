import pytest
import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.domain import Domain

geometry = Geometry2D()
a = 5
b = 10
domain = Domain(allowed_area=[(0, 0), (0, b), (a, b), (a, 0)])


def test_resize_poly():
    x_scale = 2
    y_scale = 3

    original_poly = domain.bound_poly
    resized_poly = geometry.resize_poly(original_poly, x_scale=x_scale, y_scale=y_scale)

    resized_square = geometry.get_square(resized_poly)
    original_square = geometry.get_square(original_poly)

    observed_difference = resized_square - original_square
    expected_difference = ((a*x_scale) * (b*y_scale)) - a*b

    assert np.isclose(observed_difference, expected_difference)


def test_rotate_poly():
    angle = 90

    original_poly = domain.bound_poly
    rotate_poly = geometry.rotate_poly(original_poly, angle=angle)

    expected_domain = Domain(allowed_area=[(-2.5, 2.5), (-2.5, 7.5), (7.5, 7.5), (7.5, 2.5)])
    expected_poly = expected_domain.bound_poly

    assert expected_poly == rotate_poly
