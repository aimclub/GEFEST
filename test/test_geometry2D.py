import pytest
import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.algs.geom.validation import self_intersection

geometry = Geometry2D(is_closed=True)
# marking length and width for testing polygon
poly_width = 10
poly_length = 20

# creating a testing polygons via corner points
rectangle_points = [(0, 0), (0, poly_length), (poly_width, poly_length), (poly_width, 0)]
rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(0, 0), (poly_width, poly_length), (0, poly_length)]
triangle_poly = Polygon('triangle', points=[Point(*coords) for coords in triangle_points])

incorrect_points = [(0, 0), (0, poly_length), (poly_width, poly_length), (0, poly_length), (poly_width, 0)]
incorrect_poly = Polygon('incorrect_poly', points=[Point(*coords) for coords in incorrect_points])

# creating an expected rotated polygon for testing rotate_poly() function
exp_coords = [(-poly_width / 2, poly_width / 2), (-poly_width / 2, poly_length - poly_width / 2),
              (poly_length - poly_width / 2, poly_length - poly_width / 2),
              (poly_length - poly_width / 2, poly_width / 2)]
exp_rectangle_poly = Polygon(polygon_id='expected', points=[Point(*coords) for coords in exp_coords])


def test_resize_poly():
    """Test for resize_poly function from Geometry2D class"""

    x_scale = 2
    y_scale = 3

    original_poly = rectangle_poly
    resized_poly = geometry.resize_poly(original_poly, x_scale=x_scale, y_scale=y_scale)

    resized_square = geometry.get_square(resized_poly)
    original_square = geometry.get_square(original_poly)

    observed_difference = resized_square - original_square
    expected_difference = ((poly_width * x_scale) * (poly_length * y_scale)) - poly_width * poly_length

    assert isinstance(resized_poly, Polygon)
    assert np.isclose(observed_difference, expected_difference)


@pytest.mark.parametrize('angle, expected_poly', [(90, exp_rectangle_poly), (180, rectangle_poly)])
def test_rotate_poly(angle, expected_poly):
    """Test for rotate_poly function from Geometry2D class"""

    rotate_poly = geometry.rotate_poly(rectangle_poly, angle=angle)

    rotated_coords = [tuple(coords.coords()) for coords in rotate_poly.points]
    expected_coords = [tuple(coords.coords()) for coords in expected_poly.points]

    assert set(rotated_coords).issubset(expected_coords) and len(rotated_coords) == len(expected_coords)


@pytest.mark.parametrize('figure, expected_poly', [(rectangle_poly, poly_width * poly_length),
                                                   (triangle_poly, poly_width * poly_length / 2)])
def test_get_square(figure, expected_poly):
    """Test for get_square function from Geometry2D class"""

    observed_square = geometry.get_square(figure)

    assert observed_square == expected_poly


@pytest.mark.parametrize('figure', [rectangle_poly, triangle_poly])
def test_contains_point(figure):
    """Test for get_square function from Geometry2D class"""

    expected_point = Point(1, 3)
    assert geometry.is_contain_point(figure, expected_point)

    expected_point = Point(-1, -1)
    assert not geometry.is_contain_point(figure, expected_point)


@pytest.mark.parametrize('figure_1, figure_2, expected_point',
                         [(Point(*rectangle_points[3]), rectangle_poly, Point(*rectangle_points[3]))])
def test_nearest_point(figure_1, figure_2, expected_point):
    """Test for nearest_point function from Geometry2D class"""
    observed_point = geometry.nearest_point(figure_1, figure_2)

    assert observed_point.coords() == expected_point.coords()


def test_get_convex():
    """Test for get_convex function from Geometry2D class"""
    poly_to_structure = Structure([incorrect_poly])
    assert self_intersection(poly_to_structure)

    transformed_poly = geometry.get_convex(*poly_to_structure.polygons)
    poly_to_structure = Structure([transformed_poly])
    assert not self_intersection(poly_to_structure)


def test_intersects():
    """Test for intersects function from Geometry2D class"""
    assert geometry.intersects(Structure([rectangle_poly]))


def test_distance():
    """Test for distance function from Geometry2D class"""
    dist_1 = geometry.min_distance(rectangle_poly,
                                   triangle_poly)
    assert np.isclose(dist_1, 0)
