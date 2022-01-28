import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon

geometry = Geometry2D()
a = 10
b = 20
rectangle_points = [(0,0), (0,b), (a,b), (a,0)]
rectange_poly = Polygon('rectangle', points=[Point(*coords) for coords in rectangle_points])


def test_resize_poly():
    """Test for resize_poly function from Geometry2D"""

    x_scale = 2
    y_scale = 3

    original_poly = rectange_poly
    resized_poly = geometry.resize_poly(original_poly, x_scale=x_scale, y_scale=y_scale)

    resized_square = geometry.get_square(resized_poly)
    original_square = geometry.get_square(original_poly)

    observed_difference = resized_square - original_square
    expected_difference = ((a*x_scale) * (b*y_scale)) - a*b

    assert isinstance(resized_poly, Polygon)
    assert np.isclose(observed_difference, expected_difference)


def test_rotate_poly():
    """Test for rotate_poly function from Geometry2D"""

    def getting_coords(points):
        """Subfunction for supporting to get coordinates from Point object"""

        final_coords = []
        for coords in points:
            x, y, z = coords.coords()
            final_coords.append((x,y))
        return final_coords

    def angle_90():
        """Testing rotate function with angle=90"""

        angle = 90

        rotate_poly = geometry.rotate_poly(rectange_poly, angle=angle)
        expected_poly = Polygon(polygon_id='expected', points=[Point(*coords) for coords in [(-a/2, a/2),
                                                                                            (-a/2, b-a/2),
                                                                                            (b-a/2, b-a/2),
                                                                                            (b-a/2, a/2)]])

        rotated_coords = getting_coords(rotate_poly.points)
        expected_coords = getting_coords(expected_poly.points)

        assert [False for i in zip(rotated_coords, expected_coords) if
                expected_coords != rotated_coords]

    def angle_180():
        """Testing rotate function with angle=180"""
        angle = 180

        rotate_poly = geometry.rotate_poly(rectange_poly, angle=angle)

        rotated_coords = getting_coords(rotate_poly.points)
        expected_coords = getting_coords(rectange_poly.points)

        assert [False for i in zip(rotated_coords, expected_coords) if
                expected_coords != rotated_coords]
