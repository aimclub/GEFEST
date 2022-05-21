import timeit
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gefest.core.geometry.geometry_2d import Geometry2D
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
def multi_loss(struct: Structure, expected_poly_num: int):
    num_polys = len(struct.polygons)
    penalty = 20
    loss = 0

    if len(struct.polygons) == 0:
        return None

    for poly in struct.polygons:
        quality_of_poly = area_length_ratio(poly)
        if quality_of_poly is None:
            return None
        loss += quality_of_poly
    loss = loss / len(struct.polygons)
    L = loss + penalty * abs(num_polys - expected_poly_num)

    return L


expected_poly_nums = list(range(1, 10))

fint = []

num_iters = 10
for expected_poly_num in expected_poly_nums:
    local_fint = []
    for iter in range(num_iters):
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
                        min_points_num=5,
                        is_closed=is_closed)

        task_setup = Setup(domain=domain)

        # Optimizing stage
        start = timeit.default_timer()
        result = optimize(task_setup=task_setup,
                          objective_function=partial(multi_loss,
                                                     expected_poly_num=expected_poly_num),
                          pop_size=100,
                          max_gens=100)
        optimized_structure = result.best_structure
        spend_time = timeit.default_timer() - start
        result.name = f'exp1_{expected_poly_num}_{iter}'
        result.metadata['time'] = spend_time
        result.fitness = multi_loss(optimized_structure, expected_poly_num)

        result.save(f'{result.name}.json')
        # Visualization optimized structure
        visualiser = StructVizualizer(task_setup.domain)
        plt.figure(figsize=(7, 7))

        info = {'spend_time': spend_time,
                'fitness': multi_loss(optimized_structure, expected_poly_num),
                'type': 'prediction'}
        # visualiser.plot_structure(optimized_structure, info)

        local_fint.append((info['fitness']))
        # plt.title(f'Expected structures: {expected_poly_num}')
        # plt.show()
    fint.append(local_fint)

print(fint)

sns.boxplot(x=expected_poly_nums, y=fint)
plt.show()
# plt.plot(expected_poly_nums, fint)
# plt.xlabel('Expected number of structures')
# plt.ylabel('Obtained design error')
# plt.show()
