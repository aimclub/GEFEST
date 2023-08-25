import math
from random import randint


def tournament_selection(pop, pop_size, fraction=0.1):
    """The method allows to select the best ones from whole population
    Args:
        fraction: value for separating the best part of population from another. Defaults to 0.1.
    Returns:
        The best individuals from given population. Their number is equal to ``'initial_number' * fraction``
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
