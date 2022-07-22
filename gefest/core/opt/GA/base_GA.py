import copy
import math
from random import randint

import numpy as np

from gefest.core.opt.individual import Individual
from gefest.core.opt.setup import Setup
from gefest.core.viz.struct_vizualizer import StructVizualizer


class BaseGA:
    """Genetic algorithm (GA)

    Args:
        calculate_objectives (Callable): 
    """

    def __init__(self, params, calculate_objectives,
                 evolutionary_operators, task_setup: Setup,
                 visualiser=None):

        self.params = params

        self.calculate_objectives = calculate_objectives
        self.operators = evolutionary_operators

        self.task_setup = task_setup

        self.__init_operators()
        self.__init_populations()

        self.visualiser = StructVizualizer(self.task_setup.domain)

        self.generation_number = 0

    def __init_operators(self):
        self.init_population = self.operators.init_population
        self.crossover = self.operators.crossover
        self.mutation = self.operators.mutation

    def __init_populations(self):

        gens = self.init_population(self.params.pop_size, self.task_setup.domain)
        self._pop = [Individual(genotype=gen) for gen in gens]

    class Params:
        def __init__(self, max_gens, pop_size, crossover_rate, mutation_rate, mutation_value_rate):
            self.max_gens = max_gens
            self.pop_size = pop_size
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate
            self.mutation_value_rate = mutation_value_rate

    def solution(self, verbose=True, **kwargs):
        pass

    def fitness(self):
        self.calculate_objectives(population=self._pop)
        for ind in self._pop:
            ind.fitness = ind.objectives[0]
        self._pop = [ind for ind in self._pop if ind.fitness is not None]

    def random_selection(self, group_size):
        return [self._pop[randint(0, len(self._pop) - 1)] for _ in range(group_size)]

    def tournament_selection(self, fraction=0.1):
        group_size = math.ceil(len(self._pop) * fraction)
        min_group_size = 2 if len(self._pop) > 1 else 1
        group_size = max(group_size, min_group_size)
        chosen = []
        n_iter = 0
        while len(chosen) < self.params.pop_size:
            n_iter += 1
            group = self.random_selection(group_size)
            best = min(group, key=lambda ind: ind.fitness)
            best.generation_number = self.generation_number
            if best not in chosen:
                chosen.append(best)
            elif n_iter > self.params.pop_size + 100:
                n_iter = 0
                rnd = self._pop[randint(0, len(self._pop) - 1)]
                chosen.append(rnd)
        return chosen

    def reproduce(self, selected):

        children = []
        np.random.shuffle(selected)
        for pair_index in range(0, len(selected) - 1):
            p1 = selected[pair_index]
            p2 = selected[pair_index + 1]

            child_gen = self.crossover(s1=p1.genotype, s2=p2.genotype,
                                       domain=self.task_setup.domain,
                                       rate=self.params.crossover_rate)

            child_gen = self.mutation(structure=child_gen,
                                      domain=self.task_setup.domain,
                                      rate=self.params.mutation_rate)

            if str(child_gen) != str(p1.genotype) and str(child_gen) != str(p2.genotype):
                child = Individual(genotype=copy.deepcopy(child_gen))
                child.generation_number = self.generation_number
                children.append(child)

        return children
