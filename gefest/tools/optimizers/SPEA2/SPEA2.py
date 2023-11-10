import pickle

import numpy as np
from tqdm import tqdm

from gefest.tools.optimizers.GA.GA import BaseGA


class SPEA2(BaseGA):
    """SPEA2 algorithm realization.

    In this algorithm performance is multi-dimension
    """

    def __init__(self, opt_params):
        super().__init__(opt_params)
        self.arch_size = int(self.pop_size / 2)  # Archive size
        self.archive = []  # Archive population
        self.n_criteria = 2  # Number of criteria
        self.calculate_sepa2_fitness()

    def dominate(self, idx_1: int, idx_2: int) -> bool:
        """Checks if one indivdial not worse than other by all fitness components.

        Args:
            idx_1 (int): Index of first individual.
            idx_2 (int): Index of second individual.

        Returns:
        bool: True if self._pop[idx_1] dominate self._pop[idx_2] else False
        """
        return all(
            ind1 <= ind2
            for ind1, ind2 in zip(
                self._pop[idx_1].fitness,
                self._pop[idx_2].fitness,
            )
        )

    def strength(self) -> list[int]:
        """Calculates strength for each individ in pop and arch.

        Returns:
            list[int]: strength of each individ
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

    def raw(self) -> list[int]:
        """Calculates raw for pop and arch.

        Returns:
            list[int]: Raw of each individ.
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

    def density(self) -> list[float]:
        """Calculates density.

        Returns:
            list[float]: Density of each individ in pop and arch.
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

    def calculate_sepa2_fitness(self) -> None:
        """Calulates SPEA2 fitness function."""
        self._pop = self._pop + self.archive

        raw = self.raw()
        density = self.density()
        for idx, _ in enumerate(self._pop):
            self._pop[idx].extra_characteristics['SEPA2_fitness'] = raw[idx] + density[idx]

    def environmental_selection(self) -> None:
        """Updates archive population via environmental selection procedure."""
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
        """Optimizes population."""
        for step in tqdm(range(self.n_steps)):
            # self._pop = self.selector(self._pop, self.opt_params.pop_size)
            self.environmental_selection()
            if step == self.n_steps:
                return self.archive

            self._pop = self.selector(self._pop, self.opt_params.pop_size)
            children = self.crossover(self._pop)
            mutated_children = self.mutation(children)
            extras = self.sampler(self.opt_params.extra)
            self._pop.extend(mutated_children + extras)
            self._pop = self.objectives_evaluator(self._pop)
            self.calculate_sepa2_fitness()
            self.log_dispatcher.log_pop(self._pop, str(step + 1))

        self._pop = sorted(self._pop, key=lambda x: x.fitness)
        return self._pop
