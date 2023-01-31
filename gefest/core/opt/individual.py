from copy import deepcopy
from uuid import uuid4


class Individual:
    def __init__(self, genotype):
        self.objectives = ()
        self.analytics_objectives = []
        self.fitness = None
        self.genotype = deepcopy(genotype)
        self.graph = self.genotype
        self.population_number = 0
        self.uid = str(uuid4())
