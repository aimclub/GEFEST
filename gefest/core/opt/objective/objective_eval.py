from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.opt.objective.objective import Objective
from gefest.core.utils import where


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
