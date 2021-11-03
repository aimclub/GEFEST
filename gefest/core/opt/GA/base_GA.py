import copy
import math
from random import randint

import numpy as np

import gefest.core.opt.operators.crossover
import gefest.core.opt.operators.mutation


class BaseGA:
    def __init__(self, params, calculate_objectives, evolutionary_operators,
                 visualiser=None):
        '''
         Genetic algorithm (GA)
        '''

        self.params = params

        self.calculate_objectives = calculate_objectives
        self.operators = evolutionary_operators

        self.__init_operators()
        self.__init_populations()

        self.visualiser = visualiser

        self.generation_number = 0

    def __init_operators(self):
        self.init_population = self.operators.init_population
        self.crossover = gefest.core.opt.operators.crossover.crossover
        self.mutation = gefest.core.opt.operators.mutation.mutation

    def __init_populations(self):

        gens = self.init_population(self.params.pop_size)
        self._pop = [BaseGA.Individ(genotype=gen) for gen in gens]

    class Params:
        def __init__(self, max_gens, pop_size, crossover_rate, mutation_rate, mutation_value_rate):
            self.max_gens = max_gens
            self.pop_size = pop_size
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate
            self.mutation_value_rate = mutation_value_rate

    class Individ:
        def __init__(self, genotype):
            self.objectives = ()
            self.analytics_objectives = []
            self.fitness = None
            self.genotype = copy.deepcopy(genotype)
            self.population_number = 0

    def solution(self, verbose=True, **kwargs):
        pass

    def fitness(self):

        self.calculate_objectives(population=self._pop, visualiser=self.visualiser)
        for ind in self._pop:
            ind.fitness = ind.objectives[0]

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
                print('RAND SELECTED')
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

            child_gen = self.crossover(p1.genotype, p2.genotype, self.params.crossover_rate)
            child_gen = self.mutation(child_gen, self.params.mutation_rate)
            if str(child_gen) != str(p1.genotype) and str(child_gen) != str(p2.genotype):
                child = BaseGA.Individ(genotype=copy.deepcopy(child_gen))
                child.generation_number = self.generation_number
                children.append(child)

        return children
