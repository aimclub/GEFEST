from gefest.core.opt.gen_design import design
from cases.breakwaters.configuration_de import bw_optimizer, bw_sampler
from cases.main_conf import opt_params
from cases.sensitivity_analysis.configuration_sa import (
    sa_domain,
    sa_surrogate_estimator,
)

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


# if __name__ == "__main__":
def get_structure(n_steps=opt_params.n_steps, pop_size=opt_params.pop_size):
    optimized_population = design(
        n_steps=n_steps,
        pop_size=pop_size,
        estimator=estimator,
        sampler=sampler,
        optimizer=optimizer,
        extra=True,
    )
    return optimized_population
