from abc import ABCMeta, abstractmethod

from gefest.core.geometry import Structure


class Optimizer(metaclass=ABCMeta):
    def __init__(
        self,
        logger=None,
        **kwargs,
    ) -> None:
        self.logger = logger

    @abstractmethod
    def optimize(
        self,
        n_steps: int,
        **kwargs,
    ) -> list[Structure]:
        ...
