from abc import ABCMeta, abstractmethod
from typing import Optional

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.utils import where
from gefest.tools import Estimator


class Objective(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Domain,
        estimator: Optional[Estimator] = None,
    ) -> None:
        self.domain = domain
        self.estimator = estimator

    def __call__(
        self,
        ind: Structure,
        **kwargs,
    ) -> list[Structure]:
        return self.evaluate(ind)

    @abstractmethod
    def evaluate(
        self,
        ind: Structure,
    ) -> float:
        ...


class ObjectivesEvaluator:
    def __init__(
        self,
        objectives: list[Objective],
    ) -> None:
        self.objectives = objectives

    def __call__(
        self,
        pop: list[Structure],
        **kwargs,
    ) -> list[Structure]:
        return self.set_pop_objectives(pop=pop)

    def set_pop_objectives(
        self,
        pop: list[Structure],
    ) -> list[Structure]:
        for idx in where(pop, lambda ind: len(ind.fitness) == 0):
            pop[idx] = self.eval_objectives(pop[idx])
        return sorted(pop, key=lambda x: x.fitness)

    def eval_objectives(self, ind: Structure) -> Structure:
        ind.fitness = [obj(ind) for obj in self.objectives]
        return ind
