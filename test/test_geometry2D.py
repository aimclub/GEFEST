import pytest
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.domain import Domain



def test_Geometry2d():
    geometry = Geometry2D()
    domain = Domain()

    def resize_poly_test():
        poly = domain.bound_poly()
        x_scale = 1.5
        y_scale = 1.5
        new_poly = geometry.resize_poly(poly, x_scale= x_scale,
                                              y_scale= y_scale)
        square_difference = geometry.get_square(new_poly) - geometry.get_square(poly)

        assert square_difference != 0








