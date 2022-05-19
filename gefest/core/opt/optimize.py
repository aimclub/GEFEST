from functools import partial
from typing import Callable, List

from gefest.core.opt.GA.GA import GA
from gefest.core.opt.objectives import calculate_objectives
from gefest.core.opt.operators.operators import default_operators
from gefest.core.opt.result import Result
from gefest.core.opt.setup import Setup


def optimize(task_setup: Setup, objective_function: Callable, max_gens: int, pop_size: int) -> list:
    """The wrapper object for searching optimal solution by given arguments

    Args:
        task_setup: the wrapper object that union :obj:`Domain` and a simulator of physical process
        objective_function: cost function
        max_gens: the limit number of generation in optimization process
        pop_size: the number of population size for one gemneration

    Returns:
        the best individuals from last generation after optimization
    """
    operators = default_operators()

    params = GA.Params(max_gens=max_gens, pop_size=pop_size,
                       crossover_rate=0.6, mutation_rate=0.6,
                       mutation_value_rate=[])

    _, best = GA(
        params=params,
        calculate_objectives=partial(calculate_objectives, model_func=objective_function),
        evolutionary_operators=operators, task_setup=task_setup).solution(verbose=False)

    return Result(name='result', best_structure=best.genotype,
                  metadata={'max_gens': max_gens,
                            'pop_size': pop_size},
                  fitness=best.fitness)
