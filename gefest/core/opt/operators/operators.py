from gefest.core.opt.operators.mutation import mutation
from gefest.core.opt.operators.crossover import crossover


class EvoOperators:
    def __init__(self, crossover, mutation):
        self.crossover = crossover
        self.mutation = mutation


def default_operators():
    return EvoOperators(crossover=crossover, mutation=mutation)
