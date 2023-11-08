from abc import ABCMeta, abstractmethod
from typing import Any

from gefest.core.geometry import Structure


class Strategy(metaclass=ABCMeta):
    """Abstract class for algorithm steps."""

    @abstractmethod
    def __call__(
        self,
        pop: list[Structure],
        *args: Any,
        **kwds: Any,
    ) -> list[Structure]:
        """Must implement logic of any genetic algorithm step."""
        ...
