import pytest

from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure

@pytest.fixture
def domain():
    # Create a valid domain for testing
    return Domain(name='main', allowed_area=[(0, 0), (0, 10), (10, 10), (10, 0)],
                  is_closed=False)

@pytest.fixture
def structure():
    # Create a valid structure for testing
    return Structure(polygons=[
        Polygon(points=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        Polygon(points=[Point(2, 2), Point(3, 2), Point(3, 3), Point(2, 3)])
    ])

def test_valid_structure(structure, domain):
    # Create a valid structure for testing
    structure = Structure(polygons=[
        Polygon(points=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        Polygon(points=[Point(2, 2), Point(3, 2), Point(3, 3), Point(2, 3)])
    ])

    # Check constraints on the structure
    result = check_constraints(structure, domain=domain)

    # Assert that the result is True (i.e. the structure is valid)
    assert result is True


def test_invalid_structure(structure, domain):
    # Create an invalid structure for testing
    structure.polygons.append(Polygon(points=[Point(2, 15), Point(3, 2), Point(3, 3), Point(2, 3)]))

    # Check constraints on the structure
    result = check_constraints(structure, domain=domain)

    # Assert that the result is False (i.e. the structure is invalid)
    assert result is False
