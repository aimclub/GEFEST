from enum import Enum

import numpy as np
from scipy.spatial.distance import cdist

from gefest.core.geometry.datastructs.structure import Structure


class SPEA2:
    """SPEA2 selection strategy."""

    def __init__(self, single_demention_selection, init_pop, steps, **kwargs):
        self.steps = steps
        self.step_cntr = 0
        self.arch_size = int(len(init_pop) / 2)  # Archive size
        self.archive = []  # Archive population
        self.n_criteria = 2  # Number of criteria
        self.single_demention_selection = single_demention_selection

    def __call__(self, pop, pop_size, *args, **kwargs) -> None:
        """Calulates SPEA2 fitness function."""
        pop = pop + self.archive
        raw = self.raw(pop)
        density = self.density(pop)
        for idx, _ in enumerate(pop):
            pop[idx].extra_characteristics['SEPA2_fitness'] = raw[idx] + density[idx]

        if self.step_cntr == self.steps:
            pop = self.archive
        else:
            pop = self.single_demention_selection(pop, pop_size)

        self.step_cntr += 1
        return pop

    def dominate(self, ind1: Structure, ind2: Structure) -> bool:
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
                ind1.fitness,
                ind2.fitness,
            )
        )

    def strength(self, pop) -> list[int]:
        """Calculates strength for each individ in pop and arch.

        Returns:
            list[int]: strength of each individ
        """
        strength = []

        for i in range(len(pop)):
            count = 0
            for j in range(len(pop)):
                if j == i:
                    continue

                if self.dominate(pop[i], pop[j]):
                    count += 1

            strength.append(count)

        return strength

    def raw(self, pop) -> list[int]:
        """Calculates raw for pop and arch.

        Returns:
            list[int]: Raw of each individ.
        """
        raw = []
        strength = self.strength(pop)

        for i in range(len(pop)):
            count = 0
            for j in range(len(pop)):
                if j == i:
                    continue

                if self.dominate(pop[j], pop[i]):
                    count += strength[j]

            raw.append(count)

        return raw

    def density(self, pop) -> list[float]:
        """Calculates density.

        Returns:
            list[float]: Density of each individ in pop and arch.
        """
        density = []
        k = 0

        for i in range(len(pop)):
            distance = []
            first_point = np.array(pop[i].fitness)
            for j in range(len(pop)):
                if j == i:
                    continue

                second_point = np.array(pop[j].fitness)
                distance.append(np.linalg.norm(first_point - second_point))

            sorted_dist = np.sort(distance)
            density.append(1 / (sorted_dist[k] + 2))

        return density

    def environmental_selection(self, pop) -> None:
        """Updates archive population via environmental selection procedure."""
        self.archive = [ind for ind in pop if ind.extra_characteristics['SEPA2_fitness'] < 1]

        # First case, adding remaining best individs
        if len(self.archive) < self.arch_size:
            sorted_pop = sorted(pop, key=lambda x: x.extra_characteristics['SEPA2_fitness'])
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


class MOEAD:
    """MOEA/D selection strategy.

    For details see: https://ieeexplore.ieee.org/document/4358754?arnumber=4358754
    """

    def __init__(self, single_demention_selection, init_pop, moead_n_neighbors, *args, **kwargs):
        self.ref_dirs, self.ideal, self.neighbors = self._setup(init_pop, moead_n_neighbors)
        self.single_demention_selection = single_demention_selection

    def __call__(self, pop, pop_size, **kwargs):
        """Selects best individuals."""
        pop = self._set_moead_fitness(pop)
        selected = self.single_demention_selection(pop, pop_size)
        return selected

    def _setup(self, pop, n_neighbors=2):
        ref_dirs = self._get_uniform_weight(pop, len(pop[0].fitness))
        ideal = np.min([ind.fitness for ind in pop], axis=0)
        neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[
            :,
            :n_neighbors,
        ]
        return ref_dirs, ideal, neighbors

    def _set_moead_fitness(self, pop):
        for j, ind in enumerate(pop):
            max_fun = -1.0e30
            for n in range(len(ind.fitness)):
                diff = abs(ind.fitness[n] - self.ideal[n])
                if self.ref_dirs[j][n] == 0:
                    feval = 0.0001 * diff
                else:
                    feval = diff * self.ref_dirs[j][n]

                if feval > max_fun:
                    max_fun = feval

            ind.fitness = [max_fun + len(self.neighbors[j])]

        return pop

    def _get_uniform_weight(self, pop, n_obj):
        """Sets precomputed weights.

        Precomputed weights from
        Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II
        """
        assert n_obj == 2 or n_obj == 3, 'MOEAD can handle only 2 or 3 objectives problems'
        m = len(pop)
        if n_obj == 2:
            ref_dirs = [[None for _ in range(n_obj)] for i in range(m)]
            for n in range(m):
                a = 1.0 * float(n) / (m - 1)
                ref_dirs[n][0] = a
                ref_dirs[n][1] = 1 - a
        elif n_obj == 3:
            """
            Ported from Java code written by Wudong Liu
                    (Source: http://dces.essex.ac.uk/staff/qzhang/moead/moead-java-source.zip)
            """
            m = len(pop)

            ref_dirs = []
            for i in range(m):
                for j in range(m):
                    if i + j <= m:
                        k = m - i - j
                        weight_scalars = [None] * 3
                        weight_scalars[0] = float(i) / (m)
                        weight_scalars[1] = float(j) / (m)
                        weight_scalars[2] = float(k) / (m)
                        ref_dirs.append(weight_scalars)
            # Trim number of weights to fit population size
            ref_dirs = sorted((x for x in ref_dirs), key=lambda x: sum(x), reverse=True)
            ref_dirs = ref_dirs[:m]

        return ref_dirs


class MultiObjectiveSelectionTypes(Enum):
    """Enumerates all GEFEST multi objective selectors."""

    moead = MOEAD
    spea2 = SPEA2
