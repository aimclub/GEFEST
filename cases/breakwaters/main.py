import timeit

from gefest.core.opt.gen_design import design
from cases.breakwaters.configuration_de import bw_domain
from cases.breakwaters.configuration_spea2 import bw_optimizer, bw_sampler, bw_estimator
from cases.main_conf import opt_params

# If the value is False, pretrained models will be selected
# otherwise put path to your model
opt_params.path_to_sim = False
opt_params.path_to_sur = False

# ------------
# GEFEST tools configuration
# ------------
domain, task_setup = bw_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                  points_num=opt_params.n_points,
                                                  is_closed=opt_params.is_closed)

estimator = bw_estimator.configurate_estimator(domain=domain,
                                               path_sim=opt_params.path_to_sim)

sampler = bw_sampler.configurate_sampler(domain=domain)

optimizer = bw_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
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
spend_time = timeit.default_timer() - start
print(f'spent time {spend_time} sec')
