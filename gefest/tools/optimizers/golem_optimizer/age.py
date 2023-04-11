import numpy as np

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer

from gefest.core.opt.individual import Individual
from gefest.core.opt.operators.operators import default_operators
from gefest.tools.optimizers.GA.base_GA import BaseGA


class AGE(EvoGraphOptimizer, BaseGA):
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
        #self.__init_operators()
        self.init_populations(graph_pop)
        self.init_performance(performance)

        self._evolve_population()

        population = [ind.genotype for ind in self._pop if len(ind.genotype.polygons) != 0]

        return population

    def _evolve_population(self, **kwargs):
        """ Method realizing full evolution cycle """
        self._pop = self.calculate_fitness_and_select()
        self._pop.extend(self.reproduce(self._pop))

        return self._pop

    # --------------------------------------------------
    def calculate_fitness_and_select(self):
        self.fast_nondominated_sort(self._pop)
        front1 = np.array([ind.objectives for ind in self._fronts[0]])
        ideal_point = np.min(front1, axis=0)

        # crowding distance is positive and has to be maximized
        crowd_dist = [None] * len(self._fronts)
        crowd_dist[0] = [99999] * len(front1)

        _, p, normalization = self.survival_score(front1, ideal_point)
        for i in range(0, len(self._fronts)):  # skip first front since it is normalized by survival_score
            front = np.array([ind.objectives for ind in self._fronts[i]])
            m, _ = front.shape
            front = front / normalization
            dist = 1. / self.minkowski_distances(front, ideal_point[None, :], p=p).squeeze()
            if isinstance(dist, np.float64):
                dist = [dist]
            else:
                dist = list(dist)
            crowd_dist[i] = dist

        dists = [-dist for dists in crowd_dist for dist in dists]
        ranks = [[i] * len(self._fronts[i]) for i in range(len(self._fronts))]
        ranks = [r for rank in ranks for r in rank]

        rank_dist_idx = [(i, r, d) for i, (r, d) in enumerate(zip(ranks, dists))]
        sorted_list = sorted(
            rank_dist_idx,
            key=lambda t: (t[1], t[2])
        )

        selected = [item[0] for item in sorted_list][:self.num_of_individuals]
        out_pop = [ind for pop in self._fronts for ind in pop]

        self._pop = [ind for i, ind in enumerate(out_pop) if i in selected]

        return self._pop

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

        self._fronts = [front for front in fronts if len(front) != 0]

    def survival_score(self, front, ideal_point):
        front = np.round(front, 12, out=front)
        m, n = front.shape
        crowd_dist = np.zeros(m)

        if m < n:
            p = 1
            normalization = np.max(front, axis=0)
            return crowd_dist, p, normalization

        # shift the ideal point to the origin
        front = front

        # Detect the extreme points and normalize the front
        extreme = self.find_corner_solutions(front)
        front, normalization = self.normalize(front, extreme)

        # set the distance for the extreme solutions
        crowd_dist[extreme] = np.inf
        selected = np.full(m, False)
        selected[extreme] = True

        p = self.compute_geometry(front, extreme, n)

        nn = np.linalg.norm(front, p, axis=1)
        distances = self.pairwise_distances(front, p) / nn[:, None]

        neighbors = 2
        remaining = np.arange(m)
        remaining = list(remaining[~selected])
        for i in range(m - np.sum(selected)):
            mg = np.meshgrid(np.arange(selected.shape[0])[selected], remaining, copy=False, sparse=False)
            D_mg = distances[tuple(mg)]  # avoid Numpy's future deprecation of array special indexing

            if D_mg.shape[1] > 1:
                # equivalent to mink(distances(remaining, selected),neighbors,2); in Matlab
                maxim = np.argpartition(D_mg, neighbors - 1, axis=1)[:, :neighbors]
                tmp = np.sum(np.take_along_axis(D_mg, maxim, axis=1), axis=1)
                index: int = np.argmax(tmp)
                d = tmp[index]
            else:
                index = D_mg[:, 0].argmax()
                d = D_mg[index, 0]

            best = remaining.pop(index)
            selected[best] = True
            crowd_dist[best] = d

        return crowd_dist, p, normalization

    def compute_geometry(self, front, extreme, n):
        # approximate p(norm)
        d = self.point_2_line_distance(front, np.zeros(n), np.ones(n))
        d[extreme] = np.inf
        index = np.argmin(d)

        p = np.log(n) / (np.log(1.0 / (np.mean(front[index, :]) + 0.01)) + 0.01)

        if np.isnan(p) or p <= 0.1:
            p = 1.0
        elif p > 20:
            p = 20.0  # avoid numpy underflow

        return p

    @staticmethod
    def pairwise_distances(front, p):
        m = np.shape(front)[0]
        distances = np.zeros((m, m))
        for i in range(m):
            distances[i] = np.sum(np.abs(front[i] - front) ** p, 1) ** (1 / p)

        return distances

    @staticmethod
    def minkowski_distances(A, B, p):
        m1 = np.shape(A)[0]
        m2 = np.shape(B)[0]
        distances = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                distances[i][j] = sum(np.abs(A[i] - B[j]) ** p) ** (1 / p)

        return distances

    def find_corner_solutions(self, front):
        """Return the indexes of the extreme points"""

        m, n = front.shape

        if m <= n:
            return np.arange(m)

        # let's define the axes of the n-dimensional spaces
        w = 1e-6 + np.eye(n)
        r = w.shape[0]
        indexes = np.zeros(n, dtype=int)
        selected = np.zeros(m, dtype=int)
        for i in range(r):
            dists = self.point_2_line_distance(front, np.zeros(n), w[i, :])
            dists[selected] = np.inf  # prevent already selected to be reselected
            index = np.argmin(dists)
            indexes[i] = index
            selected[index] = True
        return indexes

    @staticmethod
    def point_2_line_distance(P, A, B):
        d = np.zeros(P.shape[0])

        for i in range(P.shape[0]):
            pa = P[i] - A
            ba = B - A
            t = np.dot(pa, ba) / np.dot(ba, ba)
            d[i] = sum((pa - t * ba) ** 2)

        return d

    @staticmethod
    def normalize(front, extreme):
        m, n = front.shape

        if len(extreme) != len(np.unique(extreme, axis=0)):
            normalization = np.max(front, axis=0)
            front = front / normalization
            return front, normalization

        # Calculate the intercepts of the hyperplane constructed by the extreme
        # points and the axes

        try:
            hyperplane = np.linalg.solve(front[extreme], np.ones(n))
            if any(np.isnan(hyperplane)) or any(np.isinf(hyperplane)) or any(hyperplane < 0):
                normalization = np.max(front, axis=0)
            else:
                normalization = 1. / hyperplane
                if any(np.isnan(normalization)) or any(np.isinf(normalization)):
                    normalization = np.max(front, axis=0)
        except np.linalg.LinAlgError:
            normalization = np.max(front, axis=0)

        normalization[normalization == 0.0] = 1.0

        # Normalization
        front = front / normalization

        return front, normalization
