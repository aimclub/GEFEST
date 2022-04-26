import pytest
from gefest.core.opt.operators.mutation import mutation
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.structure.domain import Domain


domain = Domain()
x = 5
y = 5
rectangle_points = [(x, y), (x, y+5), (x+5, y+5), (x+5, y), (x, y)]
structure = Structure([Polygon(f'rectangle from {x,y}', points=[Point(*coords) for coords in rectangle_points])])


def test_mutation_true():

    for i in range(100):
        mutated_structure = mutation(structure, domain)

        mutated_points = mutated_structure.polygons[0].points
        initial_points = structure.polygons[0].points
        mutated_square = domain.geometry.get_square(mutated_structure.polygons[0])
        initial_square = domain.geometry.get_square(structure.polygons[0])

        if all([mutated_points != initial_points,
                mutated_square != initial_square]):
            assert True


def test_mutation_false():

    mutated_structure = mutation(structure, domain, rate=0.001)

    mutated_points = mutated_structure.polygons[0].points
    initial_points = structure.polygons[0].points
    mutated_square = domain.geometry.get_square(mutated_structure.polygons[0])
    initial_square = domain.geometry.get_square(structure.polygons[0])

    assert all([mutated_points == initial_points,
                mutated_square == initial_square])
