from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
import numpy as np
from gefest.core.viz.struct_vizualizer import StructVizualizer

geometry = Geometry2D()
domain = Domain(allowed_area=[(0, 0),
                              (0, 100),
                              (100, 100),
                              (100, 0),
                              (0, 0)],
                geometry=geometry,
                max_poly_num=1,
                min_poly_num=1,
                max_points_num=300,
                min_points_num=200)

task_setup = Setup(domain=domain)

def area_length_ratio(struct: Structure):
    poly = struct.polygons[0]
    area = geometry.get_square(poly)
    length = geometry.get_length(poly)
    print('number of points ' + str(len(poly.points)))

    if area == 0:
        return None

    return (1 - 4*np.pi * area / length**2)


optimized_structure, spend_time = optimize(task_setup=task_setup,
                               objective_function=area_length_ratio,
                               pop_size=150,
                               max_gens=200)

visualiser = StructVizualizer(task_setup.domain)
visualiser.plot_structure(optimized_structure, spend_time)
