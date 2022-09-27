import timeit

from gefest.core.opt.gen_design import design
from cases.microfluidic.configuration_dl import md_domain, md_sampler, md_estimator, md_optimizer
from cases.main_conf import opt_params

opt_params.is_closed = True

# ------------
# GEFEST tools configuration
# ------------

domain, task_setup = md_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                  points_num=opt_params.n_points,
                                                  is_closed=opt_params.is_closed)

estimator = md_estimator.configurate_estimator(domain=domain)
sampler = md_sampler.configurate_sampler(domain=domain)
optimizer = md_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
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
print(f'spent time {spend_time} sec')
