from copy import deepcopy

from fedot.core.optimisers.gp_comp.individual import Individual as FedotIndividual


class Individual(FedotIndividual):
    def __init__(self, genotype):
        self.fitness = None
        self.genotype = deepcopy(genotype)

    @property
    def genotype(self):
        return self.graph

    @genotype.setter
    def genotype(self, value):
        self.graph = value
