import timeit
import sys
import pickle
sys.path.append('C:/Users/user2/GEFEST')
from gefest.core.opt.gen_design import design
from gefest.core.structure.structure import get_random_structure
from gefest.tools.estimators.simulators.sound_wave.sound_interface import SoundSimulator, generate_map
from cases.sound_waves.configuration import sound_domain, sound_estimator, sound_optimizer, sound_sampler
from cases.main_conf import opt_params

# If the value is False, pretrained models will be selected
# otherwise put path to your model
opt_params.is_closed = True


# ------------
# GEFEST tools configuration
# ------------
domain, task_setup = sound_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                  points_num=opt_params.n_points,
                                                  is_closed=opt_params.is_closed)

best_structure = get_random_structure(domain)

simulator = SoundSimulator(domain)
# best_spl = simulator.estimate(best_structure)

with open('best_structure.pickle', 'wb') as handle:
    pickle.dump(best_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)      

# with open('best_spl.pickle', 'wb') as handle:
#     pickle.dump(best_spl, handle, protocol=pickle.HIGHEST_PROTOCOL)                                          

estimator = sound_estimator.configurate_estimator(domain=domain,
                                                path_best_struct='best_structure.pickle')

sampler = sound_sampler.configurate_sampler(domain=domain)

optimizer = sound_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
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

# optimized_spl = simulator.estimate(optimized_pop)

with open('optimized_structure.pickle', 'wb') as handle:
    pickle.dump(optimized_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)      

# with open('optimized_spl.pickle', 'wb') as handle:
#     pickle.dump(best_spl, handle, protocol=pickle.HIGHEST_PROTOCOL)
