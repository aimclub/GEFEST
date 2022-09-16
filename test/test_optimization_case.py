import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.opt.gen_design import design
from gefest.core.structure.structure import Structure


# Area to length ratio, circle have maximum among all figures (that`s why it`s our optima)
def area_length_ratio(poly):
    area = geometry.get_square(poly)
    length = geometry.get_length(poly)

    if area == 0:
        return None

    ratio = 1 - 4 * np.pi * area / length ** 2

    return ratio


# Adding fine for structures containing more (less) than three polygons
def multi_loss(struct: Structure):
    num = 3
    num_polys = len(struct.polygons)
    loss = 0
    for poly in struct.polygons:
        length = area_length_ratio(poly)
        if length:
            loss += length
    L = loss + 20 * abs(num_polys - num)

    return L


# Usual GEFEST procedure for initialization domain, geometry (with closed or unclosed polygons) and task_setup
is_closed = True
geometry = Geometry2D(is_closed=is_closed)
domain = Domain(allowed_area=[(0, 0),
                              (0, 300),
                              (300, 300),
                              (300, 0),
                              (0, 0)],
                geometry=geometry,
                max_poly_num=20,
                min_poly_num=1,
                max_points_num=20,
                min_points_num=4,
                is_closed=is_closed)

task_setup = Setup(domain=domain)


def test_optimization():
    """
    ::TODO::
    Change optimization to design like in synthetic examples
    """
