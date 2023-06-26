import timeit
import pickle

from gefest.core.opt.gen_design import design
from gefest.core.structure.structure import get_random_structure
from cases.main_conf import opt_params
from cases.sound_waves.configuration import (
    sound_domain,
    sound_estimator,
    sound_optimizer,
    sound_sampler,
)

# If the value is False, pretrained models will be selected
# otherwise put path to your model
opt_params.is_closed = True
opt_params.pop_size = 15
opt_params.n_steps = 20
opt_params.n_polys = 1
opt_params.n_points = 30

# ------------
# GEFEST tools configuration
# ------------
domain, task_setup = sound_domain.configurate_domain(
    poly_num=opt_params.n_polys,
    points_num=opt_params.n_points,
    is_closed=opt_params.is_closed,
)

best_structure = get_random_structure(domain)

with open("best_structure.pickle", "wb") as handle:
    pickle.dump(best_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)

estimator = sound_estimator.configurate_estimator(
    domain=domain, path_best_struct="best_structure.pickle"
)

sampler = sound_sampler.configurate_sampler(domain=domain)

optimizer = sound_optimizer.configurate_optimizer(
    pop_size=opt_params.pop_size,
    crossover_rate=opt_params.c_rate,
    mutation_rate=opt_params.m_rate,
    task_setup=task_setup,
)

# ------------
# Generative design stage
# ------------

start = timeit.default_timer()
optimized_pop = design(
    n_steps=opt_params.n_steps,
    pop_size=opt_params.pop_size,
    estimator=estimator,
    sampler=sampler,
    optimizer=optimizer,
    extra=True,
)
spend_time = timeit.default_timer() - start
print(f"spent time {spend_time} sec")

with open("optimized_structure.pickle", "wb") as handle:
    pickle.dump(optimized_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)
