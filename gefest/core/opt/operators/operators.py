from gefest.core.opt.operators.mutation import mutation
from gefest.core.opt.operators.crossover import crossover

from gefest.core.opt.operators.initial import initial_pop_random


class EvoOperators:
    def __init__(self, init_population, crossover, mutation):
        self.init_population = init_population
        self.crossover = crossover
        self.mutation = mutation


def default_operators():
    return EvoOperators(init_population=initial_pop_random, crossover=crossover, mutation=mutation)
