import pytest
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.domain import Domain


geometry = Geometry2D()
domain = Domain()

def test_resize_poly():
    a = b = 10
    x_scale = 1.5
    y_scale = 1.5
    domain = Domain(allowed_area = [(0, 0),
                                    (0, b),
                                    (a, b),
                                    (a, 0)])
    poly = domain.bound_poly()
    new_poly = geometry.resize_poly(poly, x_scale= x_scale,
                                          y_scale= y_scale)
    square_difference = geometry.get_square(new_poly) - geometry.get_square(poly)
    square_hands_calc = ((a*x_scale) * (b*y_scale)) - a*b

    assert square_difference == square_hands_calc