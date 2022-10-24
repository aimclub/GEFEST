import timeit

from gefest.core.opt.gen_design import design
from cases.heat.configuration_dl import heat_sampler, heat_estimator
from cases.main_conf import opt_params

# If the value is False, pretrained models will be selected
# otherwise put path to your model
opt_params.path_to_sampler = False
opt_params.path_to_estimator = False

# ------------
# GEFEST tools configuration
# ------------

estimator = heat_estimator.configurate_estimator(path_to_cnn=opt_params.path_to_estimator)
sampler = heat_sampler.configurate_sampler(domain=None, path_to_sampler=opt_params.path_to_sampler)
optimizer = None

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
