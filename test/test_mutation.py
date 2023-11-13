import pytest

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.operators.mutations import add_poly_mutation, drop_poly_mutation
from gefest.core.opt.operators.mutations import mutate_structure as mutation
from gefest.core.opt.operators.mutations import (
    resize_poly_mutation,
    rotate_poly_mutation,
)


def create_rectangle(x, y, dim=5):
    """Rectangle generator util."""
    rectangle_points = [(x, y), (x, y + dim), (x + dim, y + dim), (x + dim, y), (x, y)]
    rectangle_poly = Polygon(points=[Point(*coords) for coords in rectangle_points])
    return rectangle_poly


domain = Domain(
    min_poly_num=1,
    max_poly_num=3,
    allowed_area=[
        [0, 0],
        [0, 100],
        [100, 100],
        [100, 0],
        [0, 0],
    ],
)
geometry = domain.geometry
structure = Structure([create_rectangle(5, 5), create_rectangle(5, 15), create_rectangle(15, 5)])
mut_operations = [
    drop_poly_mutation,
    add_poly_mutation,
]
mutation_each_prob = [0.5, 0.5]


def test_mutation_del_add_poly():
    """Test add and delete polygon mutations."""
    count_del_poly = 0
    count_add_poly = 0
    condition = False
    start_lenghs = len(structure.polygons)
    for _ in range(100):
        mutated_structure = mutation(
            structure,
            domain,
            operations=mut_operations,
            operation_chance=0.9999,
            operations_probs=mutation_each_prob,
        )

        if len(mutated_structure.polygons) < start_lenghs:
            count_del_poly += 1
        elif len(mutated_structure.polygons) > 2:
            count_add_poly += 1

        condition = all(
            [
                count_del_poly > 0,
                count_add_poly > 0,
            ]
        )
        if condition:
            break

    assert condition


def test_mutation_not_passed():
    """Tests mutations fails."""
    mutated_structure = mutation(
        structure,
        domain,
        operations=mut_operations,
        operation_chance=0.001,
        operations_probs=mutation_each_prob,
    )

    mutated_points = mutated_structure.polygons[0].points
    initial_points = structure.polygons[0].points
    mutated_square = domain.geometry.get_square(mutated_structure.polygons[0])
    initial_square = domain.geometry.get_square(structure.polygons[0])

    assert all([mutated_points == initial_points, mutated_square == initial_square])


@pytest.mark.parametrize(
    'mut_oper, operat_prob, expected_result',
    [
        ([rotate_poly_mutation], [1], 0),
        ([resize_poly_mutation], [1], 0),
    ],
)
def test_mutation_rotate_resize_poly(mut_oper, operat_prob, expected_result):
    """Tests rotate poly operation."""
    count_rotated_poly = expected_result
    count_resize_poly = expected_result
    for _ in range(10):
        mutated_structure = mutation(
            structure,
            domain,
            operations=mut_oper,
            operation_chance=0.9999,
            operations_probs=operat_prob,
        )
        mutated_square = [
            round(geometry.get_square(poly), 2) for poly in mutated_structure.polygons
        ]
        equivalent_points = [s == m for s, m in zip(structure.polygons, mutated_structure.polygons)]
        initial_square = [geometry.get_square(poly) for poly in structure.polygons]

        compared_squares = []
        min_numb_polys = min(len(mutated_structure.polygons), len(structure.polygons))
        for idx in range(min_numb_polys):

            compared_squares.append(initial_square[idx] == mutated_square[idx])

        for ind in range(min_numb_polys):
            if not equivalent_points[ind] and compared_squares[ind]:
                count_rotated_poly += 1
            elif not compared_squares[ind] and not equivalent_points[ind]:
                count_resize_poly += 1

    if mut_oper == [rotate_poly_mutation]:
        assert count_rotated_poly > expected_result

    if mut_oper == [resize_poly_mutation]:
        assert count_resize_poly > expected_result
