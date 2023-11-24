from contextlib import nullcontext as no_exception

import pytest
from shapely.ops import unary_union

from gefest.core.geometry import Polygon, Structure
from gefest.core.geometry.datastructs.point import Point
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import get_convex_safe_area


class TestConvexSafeArea:
    """Utility tests for convex-safe selection of new points."""

    domain = Domain(
        allowed_area=[
            [0, 0],
            [0, 100],
            [100, 100],
            [100, 0],
            [0, 0],
        ],
        min_poly_num=1,
        max_poly_num=1,
        min_points_num=3,
        max_points_num=15,
        polygon_side=0.0001,
        min_dist_from_boundary=0.0001,
        geometry_is_convex=True,
        geometry_is_closed=True,
        geometry='2D',
    )

    poly_points1 = [(21, 55), (18, 44), (41, 13), (48, 13), (56, 43)]
    test_poly1 = Polygon(Point(p[0], p[1]) for p in poly_points1)
    test_structure = Structure([test_poly1])

    @pytest.mark.parametrize(
        ', '.join(
            [
                'poly',
                'domain',
                'point_left_idx',
                'point_right_idx',
                'structure',
                'poly_idx',
                'expectation',
            ],
        ),
        [
            (test_poly1, domain, 0, 1, test_structure, 0, no_exception()),
            (test_poly1, domain, 1, 2, test_structure, 0, no_exception()),
            (test_poly1, domain, 2, 3, test_structure, 0, no_exception()),
            (test_poly1, domain, 3, 4, test_structure, 0, no_exception()),
            (test_poly1, domain, 4, 0, test_structure, 0, no_exception()),
            (test_poly1, domain, 0, 2, test_structure, 0, no_exception()),
            (test_poly1, domain, 1, 3, test_structure, 0, no_exception()),
            (test_poly1, domain, 2, 4, test_structure, 0, no_exception()),
            (test_poly1, domain, 3, 0, test_structure, 0, no_exception()),
            (test_poly1, domain, 4, 1, test_structure, 0, no_exception()),
        ],
    )
    def test_convex_safe_area_saves_convexity_neighbour_indices(
        self,
        poly,
        domain,
        point_left_idx,
        point_right_idx,
        structure,
        poly_idx,
        expectation,
    ):
        """Crossing lines case."""
        with expectation:
            movment_area = get_convex_safe_area(
                poly,
                domain,
                point_left_idx,
                point_right_idx,
                structure,
                poly_idx,
            )
            s_poly = domain.geometry._poly_to_shapely_poly(poly)
            union = unary_union([s_poly, movment_area])
            assert round(union.area, 6) == round(union.convex_hull.area, 6)
            if abs(point_right_idx - point_left_idx) == 1:
                assert s_poly.intersection(movment_area).area == 0.0
