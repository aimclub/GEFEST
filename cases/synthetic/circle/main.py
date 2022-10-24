import timeit
import pickle
import matplotlib.pyplot as plt

from gefest.core.geometry.geometry_2d import create_circle
from gefest.core.structure.structure import Structure
from gefest.core.viz.struct_vizualizer import StructVizualizer

from gefest.core.opt.gen_design import design
from cases.synthetic.circle.configuration import circle_estimator, circle_sampler, circle_optimizer, circle_domain
from cases.main_conf import opt_params

opt_params.is_closed = True

# ------------
# GEFEST tools configuration
# ------------

domain, task_setup = circle_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                      points_num=opt_params.n_points,
                                                      is_closed=opt_params.is_closed)

estimator = circle_estimator.configurate_estimator(domain=domain)
sampler = circle_sampler.configurate_sampler(domain=domain)
optimizer = circle_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
                                                   crossover_rate=opt_params.c_rate,
                                                   mutation_rate=opt_params.m_rate,
                                                   task_setup=task_setup)

# ------------
# Generative design stage
# ------------

start = timeit.default_timer()
optimized_pop = design(n_steps=opt_params.n_steps,
                       pop_size=opt_params.pop_size,
                       estimator=estimator,
                       sampler=sampler,
                       optimizer=optimizer)
spend_time = timeit.default_timer() - start

# ------------
# Visualization optimized structure
# ------------

with open(f'HistoryFiles/performance_{opt_params.n_steps - 1}.pickle', 'rb') as f:
    performance = pickle.load(f)

with open(f'HistoryFiles/population_{opt_params.n_steps - 1}.pickle', 'rb') as f:
    population = pickle.load(f)

idx_of_best = performance.index(min(performance))

plt.figure(figsize=(7, 7))
visualiser = StructVizualizer(task_setup.domain)

info = {'spend_time': spend_time,
        'fitness': performance[idx_of_best],
        'type': 'prediction'}
visualiser.plot_structure(population[idx_of_best], info)

# We also add global optima for comparison with optimized solutions
true_circle = [Structure([create_circle(population[idx_of_best]), create_circle(population[idx_of_best]),
                          create_circle(population[idx_of_best])])]
info = {'spend_time': spend_time,
        'fitness': estimator.estimate(true_circle)[0],
        'type': 'true'}
visualiser.plot_structure(true_circle[0], info)

plt.show()
