from abc import ABCMeta, abstractclassmethod
from typing import Any, Iterable

from gefest.core.geometry import Structure
from gefest.core.utils import WorkersManager


class Strategy(metaclass=ABCMeta):
    """Abstract class for algorithm steps.
    Provides shared multiprocessing pool.
    """

    def __init__(self):
        self._mp = WorkersManager()

    @abstractclassmethod
    def __call__(self, pop: Iterable[Structure], *args: Any, **kwds: Any) -> Iterable[Structure]:
        """Code"""
        ...
