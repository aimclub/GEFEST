from typing import List, Optional
from gefest.core.structure.structure import Structure


class Optimizer:
    def __init__(self, optimizer):
        """
        Base optimizer class
        :param optimizer: (Object) object with method step
        """
        self.optimizer = optimizer

    def step(self, population: List[Structure], performance: List[float], n_step: Optional[int] = None):
        """
        Making one optimizing step
        :param population: (List[Structure]), input population
        :param performance: (List(float)), performance of input population obtained by estimator
        :param n_step: (Optional(int)), number of generative design step
        :return: (List[Structure]), optimized population
        """
        population = self.optimizer.step(population, performance, n_step)

        return population
