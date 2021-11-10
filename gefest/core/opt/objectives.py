from typing import List

from gefest.core.opt.analytics import EvoAnalytics
from gefest.core.opt.individual import Individual


def calculate_objectives(population: List[Individual], model_func):
    for ind_id, ind in enumerate(population):
        structure = ind.genotype
        objective = model_func(structure)
        ind.objectives = [objective]
        ind.analytics_objectives = [objective]
        idx = ind_id
        EvoAnalytics.save_cantidate(ind.population_number,
                                    ind.objectives,
                                    ind.analytics_objectives,
                                    ind.genotype,
                                    'common_dataset', idx)
