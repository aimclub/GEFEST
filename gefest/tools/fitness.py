from abc import ABCMeta, abstractmethod
from typing import Optional, Any

from gefest.tools import Estimator
from gefest.core.geometry import Structure


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

    def set_pop_fitness(self, pop: list[Structure]):
        for idx, ind in enumerate(pop):
            pop[idx].fitness = self.fitness(ind)
        return pop

    @abstractmethod
    def fitness(self, ind: Structure) -> Any:
        ...
