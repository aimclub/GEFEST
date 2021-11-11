from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure

geometry = Geometry2D()
domain = Domain(allowed_area=[(0, 0),
                              (0, 100),
                              (100, 100),
                              (100, 0),
                              (0, 0)],
                geometry=geometry,
                max_poly_num=1)

task_setup = Setup(domain=domain)

def area_length_ratio(struct: Structure):
    poly = struct.polygons[0]
    area = geometry.get_square(poly)
    length = geometry.get_length(poly)

    if area == 0:
        return None

    return (1 - 4*3.1416*area / length**2)


optimized_structure = optimize(task_setup=task_setup,
                               objective_function=area_length_ratio,
                               pop_size=30,
                               max_gens=100)
