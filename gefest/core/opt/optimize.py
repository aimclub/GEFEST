from functools import partial
from typing import Callable

from gefest.core.opt.GA.GA import GA
from gefest.core.opt.objectives import calculate_objectives
from gefest.core.opt.operators.operators import default_operators
from gefest.core.opt.setup import Setup


def optimize(task_setup: Setup, objective_function: Callable, max_gens, pop_size):
    """The object for searching optimal solution by given arguments

    Args:
        task_setup (Setup): the union of :obj:`Domain` and a simulator of physical process
        objective_function (Callable): _description_
        max_gens (int): _description_
        pop_size (int): _description_

    Returns:
        _type_: _description_
    """
    operators = default_operators()

    params = GA.Params(max_gens=max_gens, pop_size=pop_size,
                       crossover_rate=0.6, mutation_rate=0.6,
                       mutation_value_rate=[])

    _, best = GA(
        params=params,
        calculate_objectives=partial(calculate_objectives, model_func=objective_function),
        evolutionary_operators=operators, task_setup=task_setup).solution(verbose=False)

    return best.genotype
