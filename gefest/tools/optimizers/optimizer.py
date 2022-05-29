from typing import List
from gefest.core.structure.structure import Structure


class Optimizer:
    def __init__(self, optimizer):
        """
        Base optimizer class
        :param optimizer: (Object) object with method step
        """
        self.optimizer = optimizer

    def step(self, population: List[Structure], performance: List[float]):
        """
        Making one optimizing step
        :param population: (List[Structure]), input population
        :param performance: (List(float)), performance of input population
        :return: (List[Structure]), optimized population
        """
        population = self.optimizer.step(population, performance)

        return population
