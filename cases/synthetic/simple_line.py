import timeit

import matplotlib.pyplot as plt

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.analytics import EvoAnalytics
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.viz.struct_vizualizer import StructVizualizer

"""
This file contains synthetic example for open polygons
We find structure have three polygons with 20 points and 300 length.
"""


# Loss function for this task
def len_num_ration(struct: Structure):
    L = []
    p = []
    for poly in struct.polygons:
        L.append(geometry.get_length(poly))
        p.append(len(poly.points))
    length = sum(L)
    num_poly = len(struct.polygons)
    num_points = sum(p)

    ratio = abs(length - 300) + 100 * abs(num_points - 20) + 100 * abs(num_poly - 3)

    return ratio


# Usual GEFEST procedure for initialization domain, geometry (with closed or unclosed polygons) and task_setup
is_closed = False
geometry = Geometry2D(is_closed=is_closed)
domain = Domain(allowed_area=[(0, 0),
                              (0, 300),
                              (300, 300),
                              (300, 0),
                              (0, 0)],
                geometry=geometry,
                max_poly_num=10,
                min_poly_num=1,
                max_points_num=20,
                min_points_num=2,
                is_closed=is_closed)

task_setup = Setup(domain=domain)

# Start optimization
start = timeit.default_timer()
optimized_structure = optimize(task_setup=task_setup,
                               objective_function=len_num_ration,
                               pop_size=100,
                               max_gens=30)
spend_time = timeit.default_timer() - start

# Visualize optimized structure
visualiser = StructVizualizer(task_setup.domain)
plt.figure(figsize=(7, 7))

info = {'spend_time': spend_time,
        'fitness': len_num_ration(optimized_structure),
        'type': 'prediction'}
visualiser.plot_structure(optimized_structure, info)

plt.show()
EvoAnalytics.create_boxplot()
