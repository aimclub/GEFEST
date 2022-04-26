import pytest
from gefest.core.opt.operators.crossover import crossover
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.structure.domain import Domain


domain = Domain()

def create_rectangle(x, y):
    rectangle_points = [(x, y), (x, y+5), (x+5, y+5), (x+5, y), (x, y)]
    rectangle_poly = Polygon(f'rectangle from {x,y}', points=[Point(*coords) for coords in rectangle_points])
    return rectangle_poly

structure_1 = Structure([create_rectangle(5, 5), create_rectangle(5, 15)])
structure_2 = Structure([create_rectangle(15, 5), create_rectangle(15, 15)])


def test_crossover_true():

    expected_structure = [structure_1.polygons[0].points, structure_2.polygons[1].points]

    for i in range(100):
        new_structure = crossover(structure_1, structure_2, domain)
        if all([new_structure.polygons[0].points == expected_structure[0],
                new_structure.polygons[1].points == expected_structure[1]]):
            assert True



def test_crossover_false():

    new_structure = crossover(structure_1, structure_2, domain, rate=0.001)

    assert any([new_structure==structure_1, new_structure==structure_2])