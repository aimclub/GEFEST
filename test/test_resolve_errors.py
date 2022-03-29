import pytest
from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.algs.postproc.resolve_errors import *
from gefest.core.algs.geom.validation import *


# marking length and width for testing polygon
poly_width = 10
poly_length = 20

# creating a testing polygons via corner points
rectangle_points = [(-1, 40), (-1, poly_length+40), (-poly_width-10, poly_length+40), (-poly_width-10, 40)]
out_bounds_rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(1, 1), (poly_width, poly_length), (1, poly_length)]
unclosed_triangle_poly = Polygon('triangle', points=[Point(*coords) for coords in triangle_points])

incorrect_points = [(3, 3), (3, poly_length), (poly_width, poly_length), (3, poly_length), (poly_width, 5)]
incorrect_poly = Polygon('incorrect_poly', points=[Point(*coords) for coords in incorrect_points])

domain = Domain()
structure = Structure([unclosed_triangle_poly, incorrect_poly,
    out_bounds_rectangle_poly])


def test_unclosed_poly():
    input_structure = Structure([unclosed_triangle_poly])
    observed_structure = postprocess(input_structure, domain)

    assert unclosed_poly(input_structure, domain)
    assert not unclosed_poly(observed_structure, domain)

def test_self_intersection():
    input_structure = Structure([incorrect_poly])
    observed_structure = postprocess(input_structure, domain)

    assert self_intersection(input_structure)
    assert not self_intersection(observed_structure)

def test_out_of_bound():
    input_structure = Structure([out_bounds_rectangle_poly])
    observed_structure = postprocess(input_structure, domain)

    assert out_of_bound(input_structure, domain)
    assert not out_of_bound(observed_structure, domain)