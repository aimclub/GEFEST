from functools import partial
from typing import Callable, List, Union

from gefest.core.opt.GA.GA import GA
from gefest.core.opt.objectives import calculate_objectives
from gefest.core.opt.operators.operators import default_operators
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.viz.struct_vizualizer import StructVizualizer


def optimize(task_setup: Setup, objective_function: Callable, max_gens, pop_size, max_point_num, min_point_num):
    operators = default_operators()

    params = GA.Params(max_gens=max_gens, pop_size=pop_size,
                       max_point_num=max_point_num, min_point_num=min_point_num,
                       crossover_rate=0.6, mutation_rate=0.6,
                       mutation_value_rate=[])
    _, best = GA(
        params=params,
        calculate_objectives=partial(calculate_objectives, model_func=objective_function),
        evolutionary_operators=operators, task_setup=task_setup).solution(verbose=False)

    return best.genotype
