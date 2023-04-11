from functools import partial
import numpy as np
from scipy.spatial.distance import cdist

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from gefest.core.opt.individual import Individual

from gefest.core.opt.operators.operators import default_operators
from gefest.tools.optimizers.GA.base_GA import BaseGA


class MOEAD(EvoGraphOptimizer, BaseGA):
    """
    GOLEM based optimizer of GEFEST structures
    """

    def __init__(self,
                 adapter,
                 mutation_rate,
                 crossover_rate,
                 pop_size,
                 task_setup,
                 n_neighbors,
                 params):
        EvoGraphOptimizer.__init__(self, **params)

        self.params = BaseGA.Params(pop_size=pop_size,
                                    crossover_rate=crossover_rate,
                                    mutation_rate=mutation_rate,
                                    mutation_value_rate=[])
        BaseGA.__init__(self, self.params,
                        default_operators(),
                        task_setup)

        self.num_of_individuals = pop_size
        self.n_neighbors = n_neighbors
        self.task_setup = task_setup

        self.adapter = adapter
        self.golem_params = {}

    def _setup(self):
        n_obj = len(self._pop[0].objectives)
        self.initUniformWeight(n_obj)
        self.ideal = np.min([ind.objectives for ind in self._pop], axis=0)
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]
        _ = 1

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
        self._setup()

        self._evolve_population()

        population = [ind.genotype for ind in self._pop if len(ind.genotype.polygons) != 0]

        return population

    def _evolve_population(self, **kwargs):
        """ Method realizing full evolution cycle """
        self.calculate_fitness()
        self._pop = self.tournament_selection()
        self._pop.extend(self.reproduce(self._pop))

        return self._pop

    def calculate_fitness(self):
        for j, ind in enumerate(self._pop):
            maxFun = -1.0e+30
            for n in range(len(ind.objectives)):
                diff = abs(ind.objectives[n] - self.ideal[n])
                if self.ref_dirs[j][n] == 0:
                    feval = 0.0001 * diff
                else:
                    feval = diff * self.ref_dirs[j][n]

                if feval > maxFun:
                    maxFun = feval

            ind.fitness = maxFun + len(self.neighbors[j])

    def initUniformWeight(self, n_obj):
        """
        Precomputed weights from (Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II) downloaded from:
        http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar)

        """
        assert n_obj == 2 or n_obj == 3, 'MOEAD can handle only 2 or 3 objectives problems'

        m = len(self._pop)
        if n_obj == 2:
            self.ref_dirs = [[None for _ in range(n_obj)] for i in range(m)]
            for n in range(m):
                a = 1.0 * float(n) / (m - 1)
                self.ref_dirs[n][0] = a
                self.ref_dirs[n][1] = 1 - a
        elif n_obj == 3:
            """
            Ported from Java code written by Wudong Liu 
            (Source: http://dces.essex.ac.uk/staff/qzhang/moead/moead-java-source.zip)
            """
            m = len(self._pop)

            self.ref_dirs = list()
            for i in range(m):
                for j in range(m):
                    if i + j <= m:
                        k = m - i - j
                        weight_scalars = [None] * 3
                        weight_scalars[0] = float(i) / (m)
                        weight_scalars[1] = float(j) / (m)
                        weight_scalars[2] = float(k) / (m)
                        self.ref_dirs.append(weight_scalars)
            # Trim number of weights to fit population size
            self.ref_dirs = sorted((x for x in self.ref_dirs), key=lambda x: sum(x), reverse=True)
            self.ref_dirs = self.ref_dirs[:m]
