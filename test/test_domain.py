from contextlib import nullcontext as no_exception

import pytest

from gefest.core.geometry.domain import Domain


class TestDomain:
    """Domain class tests collection."""

    @pytest.mark.parametrize(
        ', '.join(
            [
                'allowed_area',
                'min_poly_num',
                'max_poly_num',
                'min_points_num',
                'max_points_num',
                'geometry_is_closed',
                'expectation',
            ]
        ),
        [
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 3, 6, True, no_exception()),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 3, 3, True, no_exception()),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 2, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 2, 2, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 2, 6, False, no_exception()),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 2, 2, False, no_exception()),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 1, 2, False, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 1, 1, False, pytest.raises(ValueError)),
            ([(1, 2), (3, 4)], 1, 2, 3, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4)], 1, 2, 3, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 2, 1, 3, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, 6, 3, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], -1, 2, 3, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], -2, -1, 3, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, -3, 6, True, pytest.raises(ValueError)),
            ([(1, 2), (3, 4), (4, 0)], 1, 2, -6, -3, True, pytest.raises(ValueError)),
        ],
    )
    def test_domain_init_validity(
        self,
        allowed_area,
        min_poly_num,
        max_poly_num,
        min_points_num,
        max_points_num,
        geometry_is_closed,
        expectation,
    ):
        """Test doamin init fails on invalid configurations."""
        with expectation:
            domain = Domain(
                allowed_area=allowed_area,
                min_poly_num=min_poly_num,
                max_poly_num=max_poly_num,
                min_points_num=min_points_num,
                max_points_num=max_points_num,
                geometry_is_closed=geometry_is_closed,
            )
            assert isinstance(domain, Domain) is True
