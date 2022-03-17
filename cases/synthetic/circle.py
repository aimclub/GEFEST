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

"""
This file contains synthetic example for closed polygons, we solve isoperimetric task. The global optimum is circle.
Additionally to loss from isoperimetric task you can add fine for number of polygons,
in this example we find three circles.
"""


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
                max_poly_num=7,
                min_poly_num=1,
                max_points_num=20,
                min_points_num=5,
                is_closed=is_closed)

task_setup = Setup(domain=domain)

# Optimizing stage
start = timeit.default_timer()
optimized_structure = optimize(task_setup=task_setup,
                               objective_function=multi_loss,
                               pop_size=100,
                               max_gens=220)
spend_time = timeit.default_timer() - start

# Visualization optimized structure
visualiser = StructVizualizer(task_setup.domain)
plt.figure(figsize=(7, 7))

info = {'spend_time': spend_time,
        'fitness': multi_loss(optimized_structure),
        'type': 'prediction'}
visualiser.plot_structure(optimized_structure, info)

# We also add global optima for comparison with optimized solutions
true_circle = Structure(create_circle(optimized_structure))
info = {'spend_time': spend_time,
        'fitness': 3 * area_length_ratio(true_circle.polygons[0]),
        'type': 'true'}
visualiser.plot_structure(true_circle, info)

plt.show()
EvoAnalytics.create_boxplot()
