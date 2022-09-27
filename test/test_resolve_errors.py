import pytest
from copy import deepcopy
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.algs.postproc.resolve_errors import *
from gefest.core.algs.geom.validation import *

# marking length and width for testing polygon
poly_width = 10
poly_length = 20

# creating a testing polygons via corner points
rectangle_points = [(-1, 40), (-1, poly_length + 40), (-poly_width - 10, poly_length + 40), (-poly_width - 10, 40)]
out_bounds_rectangle_poly = Polygon('rectangle', points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(1, 1), (poly_width, poly_length), (1, poly_length)]
unclosed_triangle_poly = Polygon('triangle', points=[Point(*coords) for coords in triangle_points])

incorrect_points = [(5, 5), (5, poly_length), (8, poly_length), (5, 5), (5, 30)]
incorrect_poly = Polygon('incorrect_poly', points=[Point(*coords) for coords in incorrect_points])

domain = Domain()


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


def test_fixed_polys():
    """
    Не понял суть теста, почему like_fixed должен уйти? А fixed появиться в poly?

    domain = Domain(fixed_points=[[[15, 30],
                                   [40, 30],
                                   [15, 40]]])
    poly_like_fixed = Polygon('like_fixed', points=[Point(15, 30), Point(40, 30), Point(15, 40)])
    input_structure = Structure([poly_like_fixed, unclosed_triangle_poly])
    observed_structure = postprocess(input_structure, domain)

    assert all([np.isclose(len(observed_structure.polygons), 2),
                'like_fixed' not in [poly.id for poly in observed_structure.polygons],
                'fixed' in [poly.id for poly in observed_structure.polygons]])
    """


def test_too_close():
    same_poly = deepcopy(unclosed_triangle_poly)
    same_poly.id = 'same_triangle'
    input_structure = Structure([unclosed_triangle_poly, same_poly])
    observed_structure = postprocess(input_structure, domain)

    print(observed_structure.polygons)

    assert np.isclose(len(observed_structure.polygons), 1)
