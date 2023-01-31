import timeit
import pickle
import matplotlib.pyplot as plt

from gefest.core.viz.struct_vizualizer import StructVizualizer

from gefest.core.opt.gen_design import design
from cases.synthetic.multiobj.configuration import multiobj_estimator, multiobj_optimizer, multiobj_domain
from cases.synthetic.line.configuration import line_sampler
from cases.main_conf import opt_params

# ------------
# GEFEST tools configuration_de
# ------------

domain, task_setup = multiobj_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                        points_num=opt_params.n_points,
                                                        is_closed=opt_params.is_closed)

estimator = multiobj_estimator.configurate_estimator(domain=domain)
sampler = line_sampler.configurate_sampler(domain=domain)
optimizer = multiobj_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
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

visualiser = StructVizualizer(task_setup.domain)
plt.figure(figsize=(7, 7))

info = {'spend_time': spend_time,
        'fitness': performance[idx_of_best],
        'type': 'prediction'}
plt.scatter(150, 150, label='target')
visualiser.plot_structure(population[idx_of_best], info)

plt.show()
