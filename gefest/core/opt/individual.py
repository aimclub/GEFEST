from copy import deepcopy


class Individual:
    def __init__(self, genotype):
        self.objectives = ()
        self.analytics_objectives = []
        self.fitness = None
        self.genotype = deepcopy(genotype)
        self.population_number = 0
