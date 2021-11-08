from typing import List

from gefest.core.opt.analytics import EvoAnalytics
from gefest.core.opt.individual import Individual


def calculate_objectives(population: List[Individual], model_func):
    for ind_id, ind in enumerate(population):
        structure = ind.genotype
        effectiveness, speed_diff, idx = model_func(structure)
        ind.objectives = [-effectiveness]  # + speed_diff / 150 + structure.size / 100000]
        ind.analytics_objectives = [-effectiveness]
        EvoAnalytics.save_cantidate(ind.population_number, ind.objectives,
                                    ind.analytics_objectives,
                                    ind.genotype,
                                    'common_dataset', idx)
