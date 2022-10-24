import pytest
from gefest.core.geometry.geometry import Geometry
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.algs.geom.validation import *

geometry = Geometry2D()
domain = Domain()
"""allowed area for Domain() is [(0, 0),
                                (0, 100),
                                (100, 100),
                                (100, 0)]
"""

poly_width = 10
poly_length = 20
# creating a testing polygons via corner points
rectangle_points = [(0, 0), (0, poly_length), (poly_width, poly_length), (poly_width, 0)]
rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(0, 0), (poly_width, poly_length), (0, poly_length)]
triangle_poly = Polygon('triangle', points=[Point(*coords) for coords in triangle_points])

out_points = [Point(x+200, y+200) for (x, y) in rectangle_points]
out_poly = Polygon('out_rectangle', points=out_points)


def test_intersection():
    intersected_points = [(1, 1), (1, poly_length), (poly_width, poly_length), (poly_width, 1)]
    intersected_rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in intersected_points])
    structure = Structure([rectangle_poly, intersected_rectangle_poly])
    assert intersection(structure, domain)

    structure = Structure([rectangle_poly, out_poly])
    assert not intersection(structure, domain)


def test_out_off_bound():
    inside_points = [(1, 1), (1, poly_length), (poly_width, poly_length), (poly_width, 1)]
    inside_rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in inside_points])
    structure = Structure([inside_rectangle_poly])
    assert not out_of_bound(structure, domain)

    structure = Structure([out_poly])
    assert out_of_bound(structure, domain)


def test_too_close():
    structure = Structure([rectangle_poly, triangle_poly])
    assert too_close(structure, domain)

    structure = Structure([rectangle_poly, out_poly])
    assert not too_close(structure, domain)


def test_self_intersection():
    incorrect_points = [(0, 0), (0, poly_length), (poly_width, poly_length), (0, poly_length), (poly_width, 0)]
    incorrect_poly = Polygon('incorrect_poly', points=[Point(*coords) for coords in incorrect_points])
    structure = Structure([incorrect_poly])
    assert self_intersection(structure)

    rebuilt_poly = geometry.get_convex(incorrect_poly)
    structure = Structure([rebuilt_poly])
    assert not self_intersection(structure)


def test_unclosed_poly():
    structure = Structure([rectangle_poly])
    assert unclosed_poly(structure, domain)

    closed_rectangle_points = [(0, 0), (0, poly_length), (poly_width, poly_length), (poly_width, 0), (0, 0)]
    closed_rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in closed_rectangle_points])
    structure = Structure([closed_rectangle_poly])
    assert not unclosed_poly(structure, domain)
