from contextlib import nullcontext as no_exception

import pytest

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.postproc.rules import Rules

geometry = Geometry2D()
prohibited_area = [(30, 30), (30, 50), (50, 50), (50, 30), (30, 30)]
prohibit_intersect = [(x, y - 5) for x, y in prohibited_area]
prohibit_intersect_2 = [(x - 5, y - 5) for x, y in prohibited_area]
domain = Domain(
    allowed_area=[
        [0, 0],
        [0, 100],
        [100, 100],
        [100, 0],
        [0, 0],
    ]
)


def poly_from_coords(coords):
    """Builds GEFEST Polygon from point tulpes."""
    return Polygon(points=[Point(*coords) for coords in coords])


rules = Rules
poly_width = 10
poly_length = 20
# creating a testing polygons via corner points
rectangle_points = [
    (0, 0),
    (0, poly_length),
    (poly_width, poly_length),
    (poly_width, 0),
]
rectangle_poly = Polygon(points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(0, 0), (poly_width, poly_length), (0, poly_length)]
triangle_poly = Polygon(points=[Point(*coords) for coords in triangle_points])

out_points = [Point(x + 200, y + 200) for (x, y) in rectangle_points]
out_poly = Polygon(points=out_points)

self_intersected_poly = [(2, 2), (3, 8), (2, 1), (1, 4), (9, 9), (2, 2)]
self_intersected_poly_open = [(2, 2), (3, 8), (2, 1), (1, 4), (9, 9)]
self_intersected_poly_2 = [(4, 4), (4, 2), (2, 2), (2, 4), (4, 4), (2, 7), (4, 7), (2, 4)]
self_intersected_poly_3 = [
    (4, 4), (4, 2), (2, 2), (2, 4), (4, 4), (2, 7), (4, 7),
    (2, 4), (4, 4), (4, 2), (2, 2), (2, 4), (4, 4)
]
self_intersected_poly_4 = [(4, 4), (4, 2), (2, 2), (2, 4), (4, 4), (4, 2), (2, 2), (2, 4), (4, 4)]
not_self_intersected_poly_closed = [(4, 4), (4, 2), (2, 2), (2, 4), (4, 4)]
not_self_intersected_poly_open = [(4, 4), (4, 2), (2, 2), (2, 4)]

out_of_bound_poly_closed = [(50, 50), (150, 150), (150, 50), (99, 0), (45, 0), (50, 50)]
out_of_bound_poly_unclosed = [(50, 50), (150, 150), (150, 50), (99, 0), (45, 0)]

structure_for_check = Structure(
    polygons=(
        [
            poly_from_coords(coords)
            for coords in [
                self_intersected_poly,
                self_intersected_poly_open,
                not_self_intersected_poly_closed,
                not_self_intersected_poly_open,
                out_of_bound_poly_closed,
                out_of_bound_poly_unclosed,
            ]
        ]
    )
)


@pytest.mark.parametrize(
    ', '.join(
        [
            'structure',
            'idx_poly_with_error',
            'domain',
            'result',
            'expectation',
        ],
    ),
    [
        (structure_for_check, 0, domain, False, no_exception()),
        (structure_for_check, 1, domain, False, no_exception()),
        (structure_for_check, 2, domain, True, no_exception()),
        (structure_for_check, 3, domain, True, no_exception()),
    ],
)
def test_self_intersection_rule(
    structure: Structure,
    idx_poly_with_error: int,
    domain: Domain,
    result: bool,
    expectation,
):
    """Test self intersection rule."""
    rule = rules.not_self_intersects.value
    with expectation:
        assert rule.validate(structure, idx_poly_with_error, domain) == result


@pytest.mark.parametrize(
    ', '.join(
        [
            'structure',
            'idx_poly_with_error',
            'domain',
            'result',
            'expectation',
        ],
    ),
    [
        (structure_for_check, 2, domain, True, no_exception()),
        (structure_for_check, 3, domain, True, no_exception()),
        (structure_for_check, 4, domain, False, no_exception()),
        (structure_for_check, 5, domain, False, no_exception()),
    ],
)
def test_out_of_bounds_rule(
    structure: Structure,
    idx_poly_with_error: int,
    domain: Domain,
    result: bool,
    expectation,
):
    """Test self intersection rule."""
    rule = rules.not_out_of_bounds.value
    with expectation:
        assert rule.validate(structure, idx_poly_with_error, domain) == result
