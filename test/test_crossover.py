from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.operators.crossovers import structure_level_crossover as crossover

domain = Domain(
    allowed_area=[
        [0, 0],
        [0, 100],
        [100, 100],
        [100, 0],
        [0, 0],
    ]
)
geometry = domain.geometry


def create_rectangle(x, y, dim=5):
    """Rectangle generation util."""
    rectangle_points = [(x, y), (x, y + dim), (x + dim, y + dim), (x + dim, y), (x, y)]
    rectangle_poly = Polygon(points=[Point(*coords) for coords in rectangle_points])
    return rectangle_poly


structure_large = Structure([create_rectangle(5, 5), create_rectangle(5, 15)])
structure_small = Structure([create_rectangle(15, 5, 3), create_rectangle(15, 15, 3)])


def test_crossover_passed():
    """Crossober valid cases."""
    expected_poly_positions = [
        structure_large.polygons[0].points,
        structure_small.polygons[1].points,
    ]
    expected_square = geometry.get_square(structure_large.polygons[0]) + geometry.get_square(
        structure_small.polygons[0]
    )

    condition = False
    for _ in range(100):
        new_structure = crossover(structure_large, structure_small, domain)[0]
        if len(new_structure.polygons) == 2:
            observed_square = geometry.get_square(new_structure.polygons[0]) + geometry.get_square(
                new_structure.polygons[1]
            )
            condition = all(
                [
                    new_structure.polygons[0].points == expected_poly_positions[0],
                    new_structure.polygons[1].points == expected_poly_positions[1],
                    observed_square == expected_square,
                ]
            )
            if condition:
                break

    assert condition
