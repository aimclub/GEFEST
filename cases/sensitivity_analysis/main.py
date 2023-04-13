import timeit
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append('C:/Users/user2/GEFEST')

from gefest.core.opt.gen_design import design
from gefest.core.utils import project_root
from cases.breakwaters.configuration_de import bw_domain
from cases.breakwaters.configuration_surrogate import  bw_estimator
from cases.sensitivity_analysis import configuration_sa #import sa_optimizer, sa_sampler, sa_surrogate_estimator
from cases.main_conf import opt_params
from gefest.core.viz.struct_vizualizer import StructVizualizer
from gefest.core.structure.structure import get_structure_from_path

# If the value is False, pretrained models will be selected
# otherwise put path to your model
opt_params.path_to_sim = False
opt_params.path_to_sur = False
opt_params.pop_size = 50
opt_params.n_steps = 5

root = project_root()


# ------------
# GEFEST tools configuration
# ------------
domain, task_setup = configuration_sa.sa_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                                   points_num=opt_params.n_points,
                                                                   is_closed=opt_params.is_closed)

estimator = configuration_sa.sa_surrogate_estimator.configurate_estimator(domain=domain,
                                                                          path_sim=opt_params.path_to_sim)

optimizer = configuration_sa.sa_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
                                                                crossover_rate=opt_params.c_rate,
                                                                mutation_rate=opt_params.m_rate,
                                                                task_setup=task_setup)

# ------------
# Generative design stage
# ------------
if __name__ == "__main__":
    for i in range(9):
        file = f'sa_structure_{i}.pkl'
        opt_params.structure_path = f'{root}/{file}'

        sampler = configuration_sa.sa_sampler.configurate_sampler(domain=domain,
                                                                  path=opt_params.structure_path)


        start = timeit.default_timer()
        optimized_pop = design(n_steps=opt_params.n_steps,
                               pop_size=opt_params.pop_size,
                               estimator=estimator,
                               sampler=sampler,
                               optimizer=optimizer,
                               extra=True)
        spend_time = round(timeit.default_timer() - start, 2)

        init_structure = get_structure_from_path(path=opt_params.structure_path)
        init_fitnes = round(estimator.estimate([init_structure])[0][0], 3)

        fitnes_history = []
        structure_history = []
        for gen_number in range(opt_params.n_steps):
            with open(f'HistoryFiles/performance_{gen_number}.pickle', 'rb') as f:
                performance = pickle.load(f)
                f.close()
            with open(f'HistoryFiles/population_{gen_number}.pickle', 'rb') as f:
                population = pickle.load(f)
                f.close()

            fitnes_history.append(min(performance)[0])
            idx_of_best = performance.index(min(performance))
            structure_history.append(population[idx_of_best])

        best_fitnes = min(fitnes_history)
        idx_of_best = fitnes_history.index(best_fitnes)
        best_structure = structure_history[idx_of_best]
        fitnes_diff = round(100 * (init_fitnes - best_fitnes)/init_fitnes, 1)

        fig, axd = plt.subplot_mosaic([['upper', 'upper'],
                                    ['lower left', 'lower right']],
                                    figsize=(10, 10), height_ratios=[1, 3])

        fig.suptitle(f'Sensitivity analysis ({opt_params.n_steps} gens, {spend_time} sec, fitnes decrease by {fitnes_diff}%)')

        init_structure.plot(color = 'r', ax=axd['lower left'], legend=True, grid=True)
        axd['lower left'].set_title(f'Initial: fitnes={init_fitnes}')
        best_structure.plot(ax=axd['lower right'], legend=True, grid=True)
        axd['lower right'].set_title(f'Processed: fitnes={round(best_fitnes, 3)}')

        axd['upper'].plot(fitnes_history, c='c', marker='o')
        axd['upper'].set_xlabel('iteration of senitivity analysis')
        axd['upper'].set_ylabel('fitnes value')

        fig.tight_layout()
        fig.savefig(f'evo_experiment_{i}.png')

