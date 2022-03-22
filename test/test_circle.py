import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure


"""
Test for synthetic case with isoperimetric task
"""

def test_fast():
    geometry = Geometry2D(is_closed=True)

    def area_length_ratio(struct: Structure):
        poly = struct.polygons[0]
        area = geometry.get_square(poly)
        length = geometry.get_length(poly)

        if area == 0:
            return None

        ratio = 1 - 4 * np.pi * area / length ** 2

        return ratio

    domain = Domain(allowed_area=[(0, 0),
                                  (0, 100),
                                  (100, 100),
                                  (100, 0),
                                  (0, 0)],
                    geometry=geometry,
                    max_poly_num=1,
                    min_poly_num=1,
                    max_points_num=30,
                    min_points_num=20,
                    is_closed=True)

    task_setup = Setup(domain=domain)

    optimized_structure = optimize(task_setup=task_setup,
                                   objective_function=area_length_ratio,
                                   pop_size=20,
                                   max_gens=1)

    assert type(optimized_structure) == Structure
