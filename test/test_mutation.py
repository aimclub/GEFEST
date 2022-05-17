import pytest
from gefest.core.opt.operators.mutation import mutation
from gefest.core.structure.structure import Structure
from gefest.core.structure.domain import Domain
from test.test_crossover import create_rectangle


domain = Domain(min_poly_num=1, max_poly_num=3)
geometry = domain.geometry
structure = Structure([create_rectangle(5, 5), create_rectangle(5, 15)])


def test_mutation_poly():
    count_del_poly = 0
    count_add_poly = 0
    count_rotated_poly = 0
    count_resize_poly = 0

    for i in range(1000):
        mutated_structure = mutation(structure, domain, rate=0.999)
        mutated_ids = [poly.id for poly in mutated_structure.polygons]
        count_mutated_points = [len(p.points) for p in mutated_structure.polygons]
        mutated_square = [geometry.get_square(poly) for poly in mutated_structure.polygons]

        if len(mutated_structure.polygons) < 2:
            count_del_poly += 1
        elif 'tmp' in mutated_ids:
            count_add_poly += 1

        count_initial_points = [len(p.points) for p in structure.polygons]
        initial_square = [geometry.get_square(poly) for poly in structure.polygons]
        compared_point_counts = []
        compared_squares = []
        min_numb_polys = min(len(mutated_structure.polygons),
                             len(structure.polygons))
        for idx in range(min_numb_polys):
            compared_point_counts.append(count_initial_points[idx] == count_mutated_points[idx])
            compared_squares.append(initial_square[idx] == mutated_square[idx])

        if any(compared_point_counts) and any(compared_squares):
            count_rotated_poly += 1
        elif not any(compared_point_counts) and not any(compared_squares):
            count_resize_poly += 1

        condition = all([count_del_poly > 0, count_add_poly > 0,
                        count_rotated_poly > 0, count_resize_poly > 0])
        if condition:
            break
    assert condition


def test_mutation_not_passed():

    mutated_structure = mutation(structure, domain, rate=0.001)

    mutated_points = mutated_structure.polygons[0].points
    initial_points = structure.polygons[0].points
    mutated_square = domain.geometry.get_square(mutated_structure.polygons[0])
    initial_square = domain.geometry.get_square(structure.polygons[0])

    assert all([mutated_points == initial_points,
                mutated_square == initial_square])
