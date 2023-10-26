import pickle

import numpy as np
from tqdm import tqdm

from gefest.tools.optimizers.GA.base_GA import BaseGA


class SPEA2(BaseGA):
    """
    SPEA2 algorithm realization
    In this algorithm performance is multi-dimension
    """

    def __init__(self, opt_params):
        super().__init__(opt_params)
        self.arch_size = int(self.pop_size / 2)  # Archive size
        self.archive = []  # Archive population
        self.n_criteria = len(self._pop[0].fitness)  # Number of criteria
        self.calculate_sepa2_fitness()
        self.environmental_selection()
        self._pop = sorted(self._pop, key=lambda x: x.extra_characteristics['SEPA2_fitness'])
        self._pop = self.selector(self._pop, self.opt_params.pop_size)

    def dominate(self, idx_1: int, idx_2: int) -> bool:
        """Check if one indivdial not worse than other
            by all fitness components.

        Args:
            idx_1 (int): Index of first individual.
            idx_2 (int): Index of second individual.

        Returns:
        bool: True if self._pop[idx_1] dominate self._pop[idx_2] else False
        """

        return all(
            [
                ind1 <= ind2
                for ind1, ind2 in zip(
                    self._pop[idx_1].fitness,
                    self._pop[idx_2].fitness,
                )
            ],
        )

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
            first_point = np.array(self._pop[i].fitness)
            for j in range(len(self._pop)):
                if j == i:
                    continue
                second_point = np.array(self._pop[j].fitness)
                distance.append(np.linalg.norm(first_point - second_point))
            sorted_dist = np.sort(distance)
            density.append(1 / (sorted_dist[k] + 2))

        return density

    def calculate_sepa2_fitness(self):
        """
        Calculating SPEA2 fitness function
        fitness = raw + density
        :return: None
        """
        self._pop = self._pop + self.archive

        raw = self.raw()
        density = self.density()
        for idx, _ in enumerate(self._pop):
            self._pop[idx].extra_characteristics['SEPA2_fitness'] = raw[idx] + density[idx]

    def environmental_selection(self):
        """
        Updating archive population via environmental selection procedure
        :return: (None)
        """
        self.archive = [ind for ind in self._pop if ind.extra_characteristics['SEPA2_fitness'] < 1]

        # First case, adding remaining best individs
        if len(self.archive) < self.arch_size:
            sorted_pop = sorted(self._pop, key=lambda x: x.extra_characteristics['SEPA2_fitness'])
            idx = 0
            while len(self.archive) != self.arch_size:
                if sorted_pop[idx].extra_characteristics['SEPA2_fitness'] >= 1:
                    self.archive.append(sorted_pop[idx])
                idx += 1

        # Second case, deleting using truncation procedure
        elif len(self.archive) > self.arch_size:
            self.archive = sorted(
                self._pop,
                key=lambda x: x.extra_characteristics['SEPA2_fitness'],
            )[: self.arch_size]

    def _save_archive(self, n):
        with open(f'HistoryFiles/archive_{n}.pickle', 'wb') as handle:
            pickle.dump(self.archive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def optimize(self):
        for step in tqdm(range(self.n_steps)):
            self._pop = self.estimator(self._pop)
            self.calculate_sepa2_fitness()
            self.environmental_selection()
            self._pop = sorted(self._pop, key=lambda x: x.extra_characteristics['SEPA2_fitness'])
            self._pop = self.selector(self._pop, self.opt_params.pop_size)
            self.log_dispatcher.log_pop(self._pop, str(step + 1))
            self.log_dispatcher.log_pop(self.archive, str(step + 1)+'_archive')
            if step == self.n_steps:
                self.log_dispatcher.log_pop(self.archive, '_archive')
                return self.archive
            self._pop = self.crossover(self._pop)
            self._pop = self.mutation(self._pop)
            self._pop.extend(self.sampler(self.opt_params.extra))


            #self._pop = sorted(self._pop, key=lambda x: x.extra_characteristics['SEPA2_fitness'])



        #self._pop = sorted(self._pop, key=lambda x: x.fitness)
        return sorted(self._pop, key=lambda x: x.extra_characteristics['SEPA2_fitness'])

    def step(self, population, performance, n_step, is_last=False):
        """
        One step of optimization procedure
        :param population: (List[Structure]) population of structures
        :param performance: (List[List]) multi dimension performance
        :param n_step: (Int) number of step
        :param is_last: (Optional(Bool)) check for last step in generative design procedure
        :return: (List[Structure]) optimized population
        """
        self.n_criteria = 2

        # Step 1, fitness assignment
        self.calculate_sepa2_fitness()

        # Step 2, environmental selection
        self.environmental_selection()

        self._save_archive(n_step)

        # Step 3, check for last step (termination)
        if is_last:
            return self.archive

        # Step 4, mating selection
        self._pop = self.selector()

        # Step 5, variation (genetic operators step)
        self._pop = self.crossover(self._pop)
        self._pop = self.mutation(self._pop)
        self._pop.extend(self.sampler(self.opt_params.extra))
        self._pop = self.estimator(self._pop)

        return population
