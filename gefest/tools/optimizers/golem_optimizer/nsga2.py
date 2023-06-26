from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from gefest.core.opt.individual import Individual

from gefest.core.opt.operators.operators import default_operators
from gefest.tools.optimizers.GA.base_GA import BaseGA
from random import randint
import random


class NSGA2(EvoGraphOptimizer, BaseGA):
    """
    GOLEM based optimizer of GEFEST structures
    """

    def __init__(self,
                 adapter,
                 mutation_rate,
                 crossover_rate,
                 pop_size,
                 task_setup,
                 params):
        EvoGraphOptimizer.__init__(self, **params)

        self.params = BaseGA.Params(pop_size=pop_size,
                                    crossover_rate=crossover_rate,
                                    mutation_rate=mutation_rate,
                                    mutation_value_rate=[])
        BaseGA.__init__(self, self.params,
                        default_operators(),
                        task_setup)

        self._pop = None
        self._fronts = None

        self.num_of_individuals = pop_size
        self.task_setup = task_setup

        self.adapter = adapter

        self.golem_params = {}

    def step(self, population, performance, n_step):
        """
        One step of optimization procedure
        :param population: (List[Structure]) population of structures
        :param performance: (List[float]) performance of the population
        :return: (List[Structure]) optimized population
        """
        # 0. Transformation List[Structure] -> Seq[OptGraph]
        # graph_pop = [self.adapter.adapt(struct) for struct in population]
        graph_pop = population

        # 1. Initializations
        self.init_populations(graph_pop)
        self.init_performance(performance)

        self._evolve_population()

        population = [ind.genotype for ind in self._pop if len(ind.genotype.polygons) != 0]

        return population

    def _evolve_population(self, **kwargs):
        """ Method realizing full evolution cycle """
        self.calculate_fitness()
        self._pop = self.tournament_selection(self._pop)
        self._pop.extend(self.reproduce(self._pop))

        return self._pop

    def calculate_fitness(self):
        self.fast_nondominated_sort(self._pop)
        for front in self._fronts:
            self.calculate_crowding_distance(front)

    # -----------------------------------------------------------------------------
    def dominates(self, ind1, ind2):
        and_condition = True
        or_condition = False
        for first, second in zip(ind1.objectives, ind2.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

    def fast_nondominated_sort(self, population):
        fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if self.dominates(individual, other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif self.dominates(other_individual, individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for individual in fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            fronts.append(temp)

        self._fronts = fronts

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    def tournament_selection(self, population):
        chosen = []
        n_iter = 0
        while len(chosen) < self.num_of_individuals:
            n_iter += 1
            best = self.__tournament(population)
            if best not in chosen:
                chosen.append(best)
            elif n_iter > self.num_of_individuals + 100:
                n_iter = 0
                rnd = population[randint(0, len(self._pop) - 1)]
                chosen.append(rnd)
        return chosen

    def __tournament(self, population):
        participants = random.sample(population, 2)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(0.9)):
                best = participant

        return best

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False
