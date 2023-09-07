from abc import ABCMeta, abstractmethod
from typing import Any

from gefest.core.geometry import Structure


class Estimator(metaclass=ABCMeta):
    def __call__(
        self,
        struct: Structure,
    ) -> Any:
        return self.estimate(struct)

    @abstractmethod
    def estimate(self, struct: Structure) -> Any:
        ...
