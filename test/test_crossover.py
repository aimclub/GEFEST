import pytest
from copy import deepcopy
from gefest.core.opt.operators.crossover import crossover
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import get_random_structure
from gefest.core.structure.structure import Structure
from gefest.core.structure.domain import Domain


def get_structure_points(structure: Structure):
    temp = []
    for poly in structure.polygons:
        temp.append([(pt.x, pt.y) for pt in poly.points])
    return temp


def test_crossover_true():
    '''testing when crossover will be passed'''

    domain = Domain(min_poly_num=2,
                    max_poly_num=4)
    structure_1 = get_random_structure(domain)
    structure_2 = get_random_structure(domain)

    new_structure = crossover(structure_1, structure_2, domain, rate=0.99)

    all_new_points = get_structure_points(new_structure)
    all_points_structure_1 = get_structure_points(structure_1)
    all_points_structure_2 = get_structure_points(structure_2)
    print(all_points_structure_1)
    print(all_points_structure_2)

    assert any([True for s1_pt in all_points_structure_1 if s1_pt in all_new_points])
    assert any([True for s2_pt in all_points_structure_2 if s2_pt in all_new_points])


# def test_crossover_false():
#     '''testing when crossover will NOT be passed'''

#     domain = Domain(min_poly_num=2,
#                     max_poly_num=4)
#     structure_1 = get_random_structure(domain)
#     structure_2 = get_random_structure(domain)

#     new_structure = crossover(structure_1, structure_2, domain, rate=0.01)

#     assert  all([s1==new for s1, new in zip(structure_1.polygons, new_structure.polygons)])
#     assert not all([s2==new for s2, new in zip(structure_2.polygons, new_structure.polygons)])
