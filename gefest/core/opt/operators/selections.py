import math
from enum import Enum
from functools import partial
from random import randint

import numpy as np

from gefest.core.geometry import Structure


def roulette_selection(
    pop: list[Structure],
    pop_size: int,
    **kwargs,
) -> list[Structure]:
    """Selects the best ones from provided population.

    Args:
        pop (list[Structure]): population
        pop_size (int): population size limit

    Returns:
        list[Structure]: best individuals from pop
    """
    _fitness = [i.fitness[0] for i in pop]
    probability = [(i / (sum(_fitness))) for i in _fitness]
    probability = [(max(probability) / i) for i in probability]
    probability = [i / sum(probability) for i in probability]

    chosen = []

    while len(chosen) < pop_size:
        chosen.append(pop[np.random.choice(a=range(len(pop)), p=probability)])

    return chosen


def tournament_selection(
    pop: list[Structure],
    pop_size: int,
    fraction: float = 0.1,
    **kwargs,
) -> list[Structure]:
    """Selects the best ones from provided population.

    Args:
        pop (list[Structure]): population
        pop_size (int): population size limit
        fraction (float, optional): best part size. Defaults to 0.1.

    Returns:
        list[Structure]: The best individuals from given population.
            Their number is equal to ``'initial_number' * fraction``
    """
    group_size = math.ceil(len(pop) * fraction)
    min_group_size = 2 if len(pop) > 1 else 1
    group_size = max(group_size, min_group_size)
    chosen = []
    n_iter = 0
    while len(chosen) < pop_size:
        n_iter += 1
        group = [pop[randint(0, len(pop) - 1)] for _ in range(group_size)]
        best = min(group, key=lambda ind: ind.fitness)
        if best not in chosen:
            chosen.append(best)
        elif n_iter > pop_size + 100:
            n_iter = 0
            rnd = pop[randint(0, len(pop) - 1)]
            chosen.append(rnd)

    return chosen


class SelectionTypes(Enum):
    """Enumerates all GEFEST selection functions."""

    roulette_selection = partial(roulette_selection)
    tournament_selection = partial(tournament_selection)
