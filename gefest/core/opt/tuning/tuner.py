from enum import Enum
from functools import partial
from typing import Callable, Union

from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.search_space import OperationParametersMapping, SearchSpace
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner

from gefest.core.algs.postproc.resolve_errors import validate
from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.tuning.uitls import VarianceGeneratorType, fitness_validation_wrap

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
        self.log_dispatcher = opt_params.log_dispatcher
        self.domain: Domain = opt_params.domain
        validator = partial(validate, rules=opt_params.postprocess_rules, domain=self.domain)
        estimator_with_validaion = partial(
            fitness_validation_wrap,
            fitness_fun=opt_params.estimator,
            validator=validator,
        )
        objective = Objective(
            quality_metrics={
                type(opt_params.estimator).__name__: estimator_with_validaion,
            },
        )
        self.tuner_type: str = opt_params.tuner_cfg.tuner_type
        self.eval_n_jobs: int = 1 if opt_params.estimator.estimator else -1
        self.objective_evaluate: ObjectiveEvaluate = ObjectiveEvaluate(objective, self.eval_n_jobs)
        self.adapter: Callable = opt_params.golem_adapter
        self.hyperopt_dist: Union[Callable, str] = opt_params.tuner_cfg.hyperopt_dist
        self.n_steps_tune: int = opt_params.tuner_cfg.n_steps_tune
        self.verbose: bool = opt_params.tuner_cfg.verbose
        self.generate_variances: VarianceGeneratorType = opt_params.tuner_cfg.variacne_generator

    def _get_tuner(
        self,
        graph: OptGraph,
        search_space: SearchSpace,
    ) -> GolemTunerType:
        return getattr(TunerType, self.tuner_type).value(
            objective_evaluate=self.objective_evaluate,
            search_space=search_space,
            adapter=self.adapter,
            iterations=self.n_steps_tune,
            n_jobs=self.eval_n_jobs,
        )

    def _generate_search_space(
        self,
        graph: OptGraph,
        sampling_variances: list[float],
    ) -> OperationParametersMapping:
        return {
            node.name: {
                param: {
                    'hyperopt-dist': self.hyperopt_dist,
                    'sampling-scope': [
                        val - sampling_variances[idx_],
                        val + sampling_variances[idx_],
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
            search_space = self._generate_search_space(
                graph,
                self.generate_variances(struct, self.domain),
            )
            tuner = self._get_tuner(graph, SearchSpace(search_space))
            tuned_structure = tuner.tune(graph=graph, show_progress=self.verbose)
            metric = tuner.obtained_metric
            tuned_structure.fitness = [metric] if not isinstance(metric, list) else metric
            tuned_objects.append(tuned_structure)
        tuned_objects = sorted(tuned_objects, key=lambda x: x.fitness)
        self.log_dispatcher.log_pop(tuned_objects, 'tuned')
        return tuned_objects
