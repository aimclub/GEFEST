from abc import ABCMeta, abstractmethod
from typing import Optional

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
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
