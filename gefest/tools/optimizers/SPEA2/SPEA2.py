import numpy as np
import pickle

from gefest.tools.optimizers.GA.base_GA import BaseGA


class SPEA2(BaseGA):
    """
    SPEA2 algorithm realization
    In this algorithm performance is multi-dimension
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arch_size = int(self.params.pop_size / 2)  # Archive size
        self.archive = []  # Archive population

        self.n_criteria = 0  # Number of criteria

    def init_performance(self, performance):
        """
        Assign each objects its performance
        :param performance: (List[List]) multi dimension performance
        :return: None
        """
        for i, ind in enumerate(self._pop):
            ind.objectives = performance[i]
        self._pop = [ind for ind in self._pop if ind.objectives is not None]

    def dominate(self, i, j):
        """
        Checking if i dominates j
        :param i: Int, index of first individ in pop + arch
        :param j: Int, index of second individ in pop + arch
        :return: (Bool) True if i dominate j else False
        """
        ind1 = self._pop[i]
        ind2 = self._pop[j]

        for i in range(self.n_criteria):
            if ind1.objectives[i] <= ind2.objectives[i]:
                continue
            else:
                return False

        return True

    def strength(self):
        """
        Calculating strength for each individ in pop and arch
        :return: (List(Int)) strength of each individ
        """
        strength = []

        for i in range(len(self._pop)):
            count = 0
            for j in range(len(self._pop)):
                if j == i:
                    continue
                if self.dominate(i, j):
                    count += 1
            strength.append(count)

        return strength

    def raw(self):
        """
        Calculating raw for pop and arch
        :return: (List(Int)) raw of each individ
        """
        raw = []
        strength = self.strength()

        for i in range(len(self._pop)):
            count = 0
            for j in range(len(self._pop)):
                if j == i:
                    continue
                if self.dominate(j, i):
                    count += strength[j]
            raw.append(count)

        return raw

    def density(self):
        """
        Calculating density
        :return: (List(float)) density of each individ in pop + arch
        """
        density = []
        k = 0

        for i in range(len(self._pop)):
            distance = []
            first_point = np.array(self._pop[i].objectives)
            for j in range(len(self._pop)):
                if j == i:
                    continue
                second_point = np.array(self._pop[j].objectives)
                distance.append(np.linalg.norm(first_point - second_point))
            sorted_dist = np.sort(distance)
            density.append(1 / (sorted_dist[k] + 2))

        return density

    def calculate_fitness(self):
        """
        Calculating SPEA2 fitness function
        fitness = raw + density
        :return: None
        """
        self._pop = self._pop + self.archive

        raw = self.raw()
        density = self.density()
        fitness = [raw[i] + density[i] for i in range(len(self._pop))]

        self.init_fitness(fitness)

    def environmental_selection(self):
        """
        Updating archive population via environmental selection procedure
        :return: (None)
        """
        self.archive = [ind for ind in self._pop if ind.fitness < 1]

        # First case, adding remaining best individs
        if len(self.archive) < self.arch_size:
            sorted_pop = sorted(self._pop, key=lambda x: x.fitness)
            idx = 0
            while len(self.archive) != self.arch_size:
                if sorted_pop[idx].fitness >= 1:
                    self.archive.append(sorted_pop[idx])
                idx += 1

        # Second case, deleting using truncation procedure
        elif len(self.archive) > self.arch_size:
            arch_obj = sorted([(ind.objectives[0], ind) for ind in self._pop], key=lambda x: x[0])[:self.arch_size]
            self.archive = [ind[1] for ind in arch_obj]

    def _save_archive(self, n):
        with open(f'HistoryFiles/archive_{n}.pickle', 'wb') as handle:
            pickle.dump(self.archive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def step(self, population, performance, n_step, is_last=False):
        """
        One step of optimization procedure
        :param population: (List[Structure]) population of structures
        :param performance: (List[List]) multi dimension performance
        :param n_step: (Int) number of step
        :param is_last: (Optional(Bool)) check for last step in generative design procedure
        :return: (List[Structure]) optimized population
        """
        assert len(performance[0]) > 1, 'Performance must be multi-dimension in SPEA2!'

        self.n_criteria = len(performance[0])

        self.init_populations(population)
        self.init_performance(performance)

        # Step 1, fitness assignment
        self.calculate_fitness()

        # Step 2, environmental selection
        self.environmental_selection()

        self._save_archive(n_step)

        # Step 3, check for last step (termination)
        if is_last:
            return self.archive

        # Step 4, mating selection
        selected = self.tournament_selection()
        self._pop = sorted(selected, key=lambda x: x.fitness)

        # Step 5, variation (genetic operators step)
        un_pop = set()
        self._pop = \
            [un_pop.add(str(ind.genotype)) or ind for ind in self._pop
             if str(ind.genotype) not in un_pop]
        self._pop = self.reproduce(self._pop)

        population = [ind.genotype for ind in self._pop]

        return population
