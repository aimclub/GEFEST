from typing import Optional, Callable, List
from gefest.core.structure.structure import Structure


class Estimator:
    def __init__(self, estimator, loss: Optional[Callable] = None):
        """
        Base estimator class, Structure -> Performance
        :param estimator: estimator with .estimate() method
        :param loss: function for minimizing, it takes estimator as argument,
                     if None estimator using as cost function
        """
        self.estimator = estimator
        self.loss = loss

    def estimate(self, population: List[Structure]):
        """
        Estimation of performance
        :param population: List(Structure) population of structures for estimating
        :return: List(Float) performance of population
        """
        performance = []
        size = len(population)

        if self.loss:
            for i in range(size):
                one_perf = self.loss(population[i], self.estimator)
                performance.append(one_perf)

        else:
            for i in range(size):
                one_perf = self.estimator.estimate(population[i])
                performance.append(one_perf)

        return performance
