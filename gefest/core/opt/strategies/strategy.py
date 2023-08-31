from abc import ABCMeta, abstractmethod
from typing import Any

from gefest.core.geometry import Structure
from gefest.core.utils import WorkersManager


class Strategy(metaclass=ABCMeta):
    """Abstract class for algorithm steps.
    Provides shared multiprocessing pool.
    """

    def __init__(self):
        self._mp = WorkersManager()

    @abstractmethod
    def __call__(self, pop: list[Structure], *args: Any, **kwds: Any,) -> list[Structure]:
        """Code"""
        ...
