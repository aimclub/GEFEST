from datetime import timedelta
from enum import Enum
from functools import partial
from typing import Callable, Union

from golem.core.optimisers.graph import OptGraph
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.search_space import OperationParametersMapping, SearchSpace
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.utilities.data_structures import ensure_wrapped_in_sequence

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.tuner_objective import GolemObjectiveWithPreValidation
from gefest.core.opt.postproc.resolve_errors import validate

VarianceGeneratorType = Callable[[Structure], list[float]]
GolemTunerType = Union[IOptTuner, OptunaTuner, SequentialTuner, SimultaneousTuner]


class TunerType(Enum):
    """Enumeration of all GOLEM tuner classes."""

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
        if opt_params.tuner_cfg is None:
            raise ValueError('TunerParams config not provided. Check cofiguration file.')

        self.log_dispatcher = opt_params.log_dispatcher
        self.domain: Domain = opt_params.domain
        self.validator = partial(validate, rules=opt_params.postprocess_rules, domain=self.domain)
        self.tuner_type: str = opt_params.tuner_cfg.tuner_type
        self.estimation_n_jobs: int = opt_params.estimation_n_jobs
        self.adapter: Callable = opt_params.golem_adapter
        self.objective = GolemObjectiveWithPreValidation(
            quality_metrics={obj.__class__.__name__: obj for obj in opt_params.objectives},
            validator=self.validator,
            adapter=self.adapter,
            is_multi_objective=len(opt_params.objectives) > 1,
        )
        self.hyperopt_distrib: Union[Callable, str] = opt_params.tuner_cfg.hyperopt_dist
        self.n_steps_tune: int = opt_params.tuner_cfg.n_steps_tune
        self.verbose: bool = opt_params.tuner_cfg.verbose
        self.timeout = timedelta(minutes=opt_params.tuner_cfg.timeout_minutes)
        self.generate_variances: VarianceGeneratorType = opt_params.tuner_cfg.variacne_generator
        self.tuner = None

    def _set_tuner(
        self,
        graph: OptGraph,
        search_space: SearchSpace,
    ) -> GolemTunerType:
        kwargs = {
            'objective_evaluate': self.objective,
            'search_space': search_space,
            'adapter': self.adapter,
            'iterations': self.n_steps_tune,
            'n_jobs': self.estimation_n_jobs,
            'timeout': self.timeout,
        }
        if self.tuner_type == 'optuna':
            kwargs['objectives_number'] = len(self.objective.metrics)

        self.tuner = getattr(TunerType, self.tuner_type).value(**kwargs)

    def _generate_search_space(
        self,
        graph: OptGraph,
        sampling_variances: list[float],
    ) -> OperationParametersMapping:
        return {
            node.name: {
                param: {
                    'hyperopt-dist': self.hyperopt_distrib,
                    'sampling-scope': [
                        sampling_variances[(idx_node * 2) + idx_coord][0],
                        sampling_variances[(idx_node * 2) + idx_coord][1],
                    ],
                    'type': 'continuous',
                }
                for idx_coord, param in enumerate(
                    list(graph.nodes[idx_node].content['params'].keys()),
                )
            }
            for idx_node, node in enumerate(graph.nodes)
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
        objects_for_tune = ensure_wrapped_in_sequence(objects_for_tune)

        tuned_objects = []
        for struct in objects_for_tune:
            graph = self.adapter.adapt(struct)
            search_space = self._generate_search_space(
                graph,
                self.generate_variances(struct, self.domain),
            )
            self._set_tuner(graph, SearchSpace(search_space))
            tuned_structures = self.tuner.tune(graph)
            metrics = self.tuner.obtained_metric

            tuned_structures = ensure_wrapped_in_sequence(tuned_structures)
            metrics = ensure_wrapped_in_sequence(self.tuner.obtained_metric)

            for idx_, _ in enumerate(tuned_structures):
                tuned_structures[idx_].fitness = ensure_wrapped_in_sequence(metrics[idx_])

        tuned_objects.extend(tuned_structures)
        tuned_objects = sorted(tuned_objects, key=lambda x: x.fitness)
        self.log_dispatcher.log_pop(tuned_objects, 'tuned')
        return tuned_objects
