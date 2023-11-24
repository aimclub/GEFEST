from abc import ABCMeta, abstractmethod
from typing import Any

from gefest.core.geometry import Structure


class Estimator(metaclass=ABCMeta):
    """Interface for estimation backends, e.g. physical simulators, neural networks."""

    def __call__(
        self,
        struct: Structure,
    ) -> Any:
        """Incapsulates estimate method call for simler estimator usage."""
        return self.estimate(struct)

    @abstractmethod
    def estimate(self, struct: Structure) -> Any:
        """Must implemet logic of estimation."""
        ...
