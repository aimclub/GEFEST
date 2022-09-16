import timeit
import argparse

from gefest.core.opt.gen_design import design
from cases.breakwaters.configuration_de import bw_domain
from cases.breakwaters.configuration_spea2 import bw_optimizer, bw_sampler, bw_estimator

parser = argparse.ArgumentParser()
parser.add_argument("--pop_size", type=int, default=30, help='number of individs in population')
parser.add_argument("--n_steps", type=int, default=80, help='number of generative design steps')
parser.add_argument('--n_polys', type=int, default=5, help='maximum number of polygons in structure')
parser.add_argument('--n_points', type=int, default=15, help='maximum number of points in polygon')
parser.add_argument('--c_rate', type=float, default=0.6, help='crossover rate')
parser.add_argument('--m_rate', type=float, default=0.6, help='mutation rate')
parser.add_argument('--is_closed', type=bool, default=False, help='type of polygon')
parser.add_argument('--path_to_sim', type=str, default=False, help='path to physical simulator')
parser.add_argument('--path_to_sur', type=str, default=False, help='path to surrogate model')
opt = parser.parse_args()


# ------------
# GEFEST tools configuration
# ------------
domain, task_setup = bw_domain.configurate_domain(poly_num=opt.n_polys,
                                                  points_num=opt.n_points,
                                                  is_closed=opt.is_closed)

estimator = bw_estimator.configurate_estimator(domain=domain,
                                               path_sim=opt.path_to_sim)

sampler = bw_sampler.configurate_sampler(domain=domain)

optimizer = bw_optimizer.configurate_optimizer(pop_size=opt.pop_size,
                                               crossover_rate=opt.c_rate,
                                               mutation_rate=opt.m_rate,
                                               task_setup=task_setup)

# ------------
# Generative design stage
# ------------

start = timeit.default_timer()
optimized_pop = design(n_steps=opt.n_steps,
                       pop_size=opt.pop_size,
                       estimator=estimator,
                       sampler=sampler,
                       optimizer=optimizer,
                       extra=True)
spend_time = timeit.default_timer() - start
print(f'spent time {spend_time} sec')

"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

domain, task_setup = bw_domain.configurate_domain(poly_num=opt.n_polys,
                                                  points_num=opt.n_points,
                                                  is_closed=opt.is_closed)
est = bw_estimator.configurate_estimator(domain=domain,
                                         path_sim=opt.path_to_sim)
for i in range(140):
    with open(f'BW_SURR_2/archive_{i}.pickle', 'rb') as f:
        data_new = pickle.load(f)
    plt.figure(i)
    hs = [d.objectives[1] for d in data_new]
    idx = np.argmin(hs)
    data = data_new[idx]
    targets = np.array([[49, 26], [11, 37], [5, 60]])

    Z, hs = est.estimate(data.genotype)
    Z = np.clip(Z, 0, 2)

    polygons = data.genotype.polygons
    for poly in polygons:
        x = np.array([p.x for p in poly.points]) / 2075 * 83
        y = np.array([p.y for p in poly.points]) / 1450 * 58
        plt.plot(x, y, '-o', color='blue', linewidth=3)
    plt.imshow(Z, cmap='OrRd')
    plt.gca().invert_yaxis()
    #plt.colorbar()

    X = targets[:, 1]
    Y = targets[:, 0]
    plt.scatter(X, Y, c='green', marker='s')
    plt.title(f'Epoch {i}, performance {round(np.min(hs), 4)}')
    plt.axis('off')

    plt.savefig(f'imgs_for_gif/{i}.png', bbox_inches='tight', pad_inches=0)
"""