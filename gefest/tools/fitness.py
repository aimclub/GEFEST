from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from loguru import logger

from gefest.core.geometry import Structure
from gefest.core.utils import where
from gefest.tools import Estimator


class Fitness(metaclass=ABCMeta):
    def __init__(
        self,
        estimator: Optional[Estimator] = None,
    ) -> None:
        self.estimator = estimator

    def __call__(
        self,
        pop: list[Structure],
        **kwargs,
    ) -> list[Structure]:
        return self.set_pop_fitness(pop=pop)

    @logger.catch
    def set_pop_fitness(self, pop: list[Structure]):
        for idx in where(pop, lambda ind: len(ind.fitness) == 0):
            pop[idx].fitness = self.fitness(pop[idx])
        return pop

    @abstractmethod
    def fitness(self, ind: Structure) -> Any:
        ...
