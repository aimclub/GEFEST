from typing import List, Union

from gefest.core.opt.GA.GA import GA
from gefest.core.opt.objectives import (calculate_objectives)
from gefest.core.opt.operators.operators import default_operators
from gefest.core.structure.domain import Domain


def optimize(domain: Union[Domain, List[Domain]], max_gens=300, pop_size=300):
    operators = default_operators()
    results = []

    params = GA.Params(max_gens=max_gens, pop_size=pop_size,
                       crossover_rate=0.6, mutation_rate=0.6,
                       mutation_value_rate=[])
    _, best = GA(
        params=params,
        calculate_objectives=calculate_objectives,
        evolutionary_operators=operators).solution(verbose=False)

    results = [best]

    return results
