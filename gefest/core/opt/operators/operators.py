from gefest.core.opt.operators.mutation import mutation
from gefest.core.opt.operators.crossover import crossover
from gefest.core.opt.operators.sensitivity_methods import sa_mutation


class EvoOperators:
    def __init__(self, crossover, mutation):
        self.crossover = crossover
        self.mutation = mutation


def default_operators():
    return EvoOperators(crossover=crossover, mutation=mutation)

def sensitivity_operators():
    return EvoOperators(crossover=crossover, mutation=sa_mutation)
