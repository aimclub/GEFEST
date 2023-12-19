from typing import Union

from golem.utilities.data_structures import ensure_wrapped_in_sequence

from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.opt.objective.objective import Objective
from gefest.core.utils import where
from gefest.core.utils.parallel_manager import BaseParallelDispatcher


class ObjectivesEvaluator:
    """Implements objecives evaluation procedure."""

    def __init__(
        self,
        objectives: list[Objective],
        n_jobs=None,
    ) -> None:
        self.objectives = objectives
        if n_jobs in (0, 1):
            self._pm = None
        else:
            self._pm = BaseParallelDispatcher(n_jobs)

    def __call__(
        self,
        pop: Union[list[Structure], Structure],
        **kwargs,
    ) -> list[Structure]:
        """Calls objectives evaluation."""
        pop = ensure_wrapped_in_sequence(pop)

        return self.set_pop_objectives(pop=pop)

    def set_pop_objectives(
        self,
        pop: list[Structure],
    ) -> list[Structure]:
        """Evaluates objectives for whole population."""
        idxs_to_eval = where(pop, lambda ind: len(ind.fitness) == 0)
        individuals_to_eval = [pop[idx] for idx in idxs_to_eval]
        if self._pm:
            evaluated_individuals = self._pm.exec_parallel(
                func=self.eval_objectives,
                arguments=[(ind, self.objectives) for ind in individuals_to_eval],
                use=True,
                flatten=False,
            )
            for idx, evaluated_ind in zip(idxs_to_eval, evaluated_individuals):
                pop[idx] = evaluated_ind
        else:
            for idx in where(pop, lambda ind: len(ind.fitness) == 0):
                pop[idx] = self.eval_objectives(pop[idx], self.objectives)

        return sorted(pop, key=lambda x: x.fitness)

    def eval_objectives(self, ind: Structure, objectives) -> Structure:
        """Evaluates objectives."""
        ind.fitness = [obj(ind) for obj in objectives]
        return ind
