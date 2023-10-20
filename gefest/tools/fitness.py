from abc import ABCMeta, abstractmethod
from typing import Optional

from loguru import logger

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.utils import where
from gefest.tools import Estimator


class Fitness(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Domain,
        estimator: Optional[Estimator] = None,
    ) -> None:
        self.domain = domain
        self.estimator = estimator

    def __call__(
        self,
        pop: list[Structure],
        **kwargs,
    ) -> list[Structure]:
        return self.set_pop_fitness(pop=pop)

    @logger.catch
    def set_pop_fitness(
        self,
        pop: list[Structure],
    ) -> list[Structure]:
        for idx in where(pop, lambda ind: len(ind.fitness) == 0):
            fit = self.fitness(pop[idx])
            if not isinstance(fit, list):
                fit = [fit]
            pop[idx].fitness = fit
        return sorted(pop, key=lambda x: [x.fitness[i] for i in range(len(x.fitness))])

    @abstractmethod
    def fitness(self, ind: Structure) -> float:
        ...
