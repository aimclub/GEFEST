import pytest
from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.algs.geom.validation import *

domain = Domain()
geometry = Geometry()
"""allowed area for Domain() is [(0, 0),
                                (0, 100),
                                (100, 100),
                                (100, 0)]
"""
poly_width = -10
poly_length = -20

# creating a testing polygons via corner points
rectangle_points = [(0, 0), (0, poly_length), (poly_width, poly_length), (poly_width, 0)]
rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(0, 0), (poly_width, poly_length), (0, poly_length)]
triangle_poly = Polygon('triangle', points=[Point(*coords) for coords in triangle_points])

structure = Structure([rectangle_poly, triangle_poly])


def test_intersection():
    assert intersection(structure, geometry)


def test_out_off_bound():
    assert out_of_bound(structure, domain)


def test_too_close():
    assert too_close(structure, domain)


def test_self_intersection():
    assert not self_intersection(structure)


def test_unclosed_poly():
    assert unclosed_poly(structure, domain)
