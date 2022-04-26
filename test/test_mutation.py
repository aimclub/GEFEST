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
        if all([mutated_structure.polygons[0].points != structure.polygons[0].points,
                domain.geometry.get_square(mutated_structure.polygons[0]) != domain.geometry.get_square(structure.polygons[0])
                ]):
            assert True


def test_mutation_false():

    mutated_structure = mutation(structure, domain, rate=0.01)
    assert all([mutated_structure.polygons[0].points == structure.polygons[0].points,
                domain.geometry.get_square(mutated_structure.polygons[0]) == domain.geometry.get_square(structure.polygons[0])
                ])
