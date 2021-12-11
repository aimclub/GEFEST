import timeit

import matplotlib.pyplot as plt
import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D, create_circle
from gefest.core.opt.analytics import EvoAnalytics
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.viz.struct_vizualizer import StructVizualizer

geometry = Geometry2D()
domain = Domain(allowed_area=[(0, 0),
                              (0, 300),
                              (300, 300),
                              (300, 0),
                              (0, 0)],
                geometry=geometry,
                max_poly_num=1,
                min_poly_num=1,
                max_points_num=40,
                min_points_num=30)

task_setup = Setup(domain=domain)


def area_length_ratio(struct: Structure):
    poly = struct.polygons[0]
    area = geometry.get_square(poly)
    length = geometry.get_length(poly)

    if area == 0:
        return None

    ratio = 1 - 4 * np.pi * area / length ** 2

    return ratio


start = timeit.default_timer()
optimized_structure = optimize(task_setup=task_setup,
                               objective_function=area_length_ratio,
                               pop_size=100,
                               max_gens=20)
spend_time = timeit.default_timer() - start

visualiser = StructVizualizer(task_setup.domain)
plt.figure(figsize=(7, 7))

info = {'spend_time': spend_time,
        'fitness': area_length_ratio(optimized_structure),
        'type': 'prediction'}
visualiser.plot_structure(optimized_structure, info)

true_circle = Structure(create_circle(optimized_structure))
info = {'spend_time': spend_time,
        'fitness': area_length_ratio(true_circle),
        'type': 'true'}
visualiser.plot_structure(true_circle, info)

plt.show()
EvoAnalytics.create_boxplot()
