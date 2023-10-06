from enum import Enum
from typing import Union

from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.search_space import OperationParametersMapping, SearchSpace
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.geometry import Structure

GolemTunerType = Union[IOptTuner, OptunaTuner, SequentialTuner, SimultaneousTuner]


class TunerType(Enum):
    iopt = IOptTuner
    optuna = OptunaTuner
    sequential = SequentialTuner
    simulataneous = SimultaneousTuner


class GolemTuner:
    """Wrap for GOLEM tuners.
    Provides interface for tuning stucture points coordinates.
    For more details about tuners see:
    https://thegolem.readthedocs.io/en/latest/api/tuning.html
    https://fedot.readthedocs.io/en/latest/advanced/hyperparameters_tuning.html
    """

    def __init__(
        self,
        opt_params: OptimizationParams,
        **kwargs,
    ) -> None:

        fitness_function = opt_params.estimator
        objective = Objective(
            quality_metrics={
                type(fitness_function).__name__: fitness_function.fitness,
            },
        )
        self.tuner_type = opt_params.tuner_cfg.tuner_type
        self.eval_n_jobs = 1 if fitness_function.estimator else -1
        self.objective_evaluate = ObjectiveEvaluate(objective, self.eval_n_jobs)
        self.adapter = opt_params.golem_adapter
        self.sampling_variance = opt_params.tuner_cfg.sampling_variance
        self.hyperopt_dist = opt_params.tuner_cfg.hyperopt_dist
        self.n_steps_tune = opt_params.tuner_cfg.n_steps_tune
        self.verbose = opt_params.tuner_cfg.verbose

    def _get_tuner(
        self,
        graph: OptGraph,
    ) -> GolemTunerType:
        return getattr(TunerType, self.tuner_type).value(
            objective_evaluate=self.objective_evaluate,
            search_space=SearchSpace(self._generate_search_space(graph)),
            adapter=self.adapter,
            iterations=self.n_steps_tune,
            n_jobs=self.eval_n_jobs,
        )

    def _generate_search_space(
        self,
        graph: OptGraph,
    ) -> OperationParametersMapping:
        return {
            node.name: {
                param: {
                    'hyperopt-dist': self.hyperopt_dist,
                    'sampling-scope': [
                        val - self.sampling_variance,
                        val + self.sampling_variance,
                    ],
                    'type': 'continuous',
                }
                for param, val in zip(
                    list(graph.nodes[idx_].content['params'].keys()),
                    list(graph.nodes[idx_].content['params'].values()),
                )
            }
            for idx_, node in enumerate(graph.nodes)
        }

    def tune(
        self,
        objects_for_tune: Union[Structure, list[Structure]],
    ) -> list[Structure]:
        """Tunes coordinstes for all points in each provided structure.

        Args:
            objects_for_tune (Union[Structure, list[Structure]]): Structures for tune.

        Returns:
            list[Structure]: Tuned structures.
        """
        if isinstance(objects_for_tune, Structure):
            objects_for_tune = [objects_for_tune]
        tuned_objects = []
        for struct in objects_for_tune:
            graph = self.adapter.adapt(struct)
            tuner = self._get_tuner(graph)
            tuned_structure = tuner.tune(graph=graph, show_progress=self.verbose)
            tuned_structure.fitness = tuner.obtained_metric
            tuned_objects.append(tuned_structure)
        return tuned_objects
