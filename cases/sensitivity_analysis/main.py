import timeit
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append('C:/Users/user2/GEFEST')

from gefest.core.opt.gen_design import design
from gefest.core.utils import project_root
from cases.breakwaters.configuration_de import bw_domain
from cases.sensitivity_analysis.configuration_sa import sa_optimizer, sa_sampler, sa_estimator
from cases.main_conf import opt_params
from gefest.core.viz.struct_vizualizer import StructVizualizer
from gefest.core.structure.structure import get_structure_from_path

# If the value is False, pretrained models will be selected
# otherwise put path to your model
opt_params.path_to_sim = False
opt_params.path_to_sur = False
opt_params.pop_size = 25
opt_params.n_steps = 20


# ------------
# GEFEST tools configuration
# ------------
domain, task_setup = bw_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                  points_num=opt_params.n_points,
                                                  is_closed=opt_params.is_closed)

estimator = sa_estimator.configurate_estimator(domain=domain,
                                               path_sim=opt_params.path_to_sim)

sampler = sa_sampler.configurate_sampler(domain=domain,
                                         path=opt_params.structure_path)

optimizer = sa_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
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
                       optimizer=optimizer,
                       extra=True)
spend_time = round(timeit.default_timer() - start, 2)
print(f'spent time {spend_time} sec')

path_init_structure = 'optimized.pkl'
init_structure = get_structure_from_path(path=path_init_structure)

with open(f'HistoryFiles/performance_{opt_params.n_steps - 1}.pickle', 'rb') as f:
    performance = pickle.load(f)

with open(f'HistoryFiles/population_{opt_params.n_steps - 1}.pickle', 'rb') as f:
    population = pickle.load(f)


fig, axd = plt.subplot_mosaic([['upper', 'upper'],
                               ['lower left', 'lower right']],
                               figsize=(10, 10), height_ratios=[1, 3])

idx_of_best = performance.index(min(performance))
best_structure = population[idx_of_best]
fitnes_1 = [fit_1 for fit_1, fit_2 in performance]
# fitnes_2 = [fit_2 for fit_1, fit_2 in performance]
# visualiser = StructVizualizer(task_setup.domain)
# plt.figure(figsize=(7, 7))

info = {'spend_time': round(spend_time, 1),
        'fitness': performance[idx_of_best][0]}
# visualiser.plot_structure(population[idx_of_best], info)


fig.suptitle(f'Sensitivity analysis ({opt_params.n_steps} gens, {spend_time} sec)')

init_structure.plot(color = 'r', ax=axd['lower left'], legend=True)
axd['lower left'].set_title('Initial structure')
best_structure.plot(ax=axd['lower right'], legend=True)
axd['lower right'].set_title('Processed structure')

axd['upper'].plot(fitnes_1, c='c')
axd['upper'].set_xlabel('iteration of senitivity analysis')
axd['upper'].set_ylabel('fitnes value')

fig.tight_layout()
fig.savefig('evo_experiment_moving.png')

