from gefest.core.algs.sensitivity.sa_core import SA, report_viz
from gefest.core.opt.gen_design import design
from cases.breakwaters.configuration_de import bw_optimizer, bw_sampler
from cases.main_conf import opt_params
from cases.sensitivity_analysis.configuration_sa import (
    sa_domain,
    sa_surrogate_estimator,
)

# ------------
# Generative design stage
# ------------
# Configurations for evolutionary optimization
opt_params.path_to_sim = False
opt_params.path_to_sur = False
opt_params.pop_size = 20
opt_params.n_steps = 50

domain, task_setup = sa_domain.configurate_domain(
    poly_num=opt_params.n_polys,
    points_num=opt_params.n_points,
    is_closed=opt_params.is_closed,
)

estimator = sa_surrogate_estimator.configurate_estimator(
    domain=domain, path_sim=opt_params.path_to_sim
)

sampler = bw_sampler.configurate_sampler(domain=domain)

optimizer = bw_optimizer.configurate_optimizer(
    pop_size=opt_params.pop_size,
    crossover_rate=opt_params.c_rate,
    mutation_rate=opt_params.m_rate,
    task_setup=task_setup,
)


optimized_structure = design(
    n_steps=opt_params.n_steps,
    pop_size=opt_params.pop_size,
    estimator=estimator,
    sampler=sampler,
    optimizer=optimizer,
    extra=True,
)


# ------------
# Sensitivity-based optimization
# ------------

sens_optimizer = SA(
    optimized_pop=optimized_structure, estimator=estimator, domain=domain
)

# For receiving only the improved structure
# improved_structure = sens_optimizer.get_improved_structure

# For receiving full history of optimization and further visualization
sens_results = sens_optimizer.analysis()

# save report image to the main directory of lib
report_viz(analysis_result=sens_results)
