from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Union

from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from pydantic import BaseModel, ConfigDict, model_validator

from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.core.opt.objective.objective import Objective
from gefest.core.opt.operators.crossovers import CrossoverTypes, panmixis
from gefest.core.opt.operators.multiobjective_selections import (
    MultiObjectiveSelectionTypes,
)
from gefest.core.opt.operators.mutations import MutationTypes
from gefest.core.opt.operators.selections import SelectionTypes
from gefest.core.opt.postproc.resolve_errors import Postrocessor
from gefest.core.opt.postproc.rules import PolygonRule, Rules, StructureRule
from gefest.core.utils.logger import LogDispatcher
from gefest.tools.optimizers import OptimizerTypes
from gefest.tools.samplers.standard.standard import StandardSampler

ValidRules = Enum(
    'ValidRules',
    ((value, value) for value in [r.name for r in Rules]),
    type=str,
)

ValidMutations = Enum(
    'ValidMutations',
    ((value, value) for value in [r.name for r in MutationTypes]),
    type=str,
)

ValidCrossovers = Enum(
    'ValidCrossovers',
    ((value, value) for value in [r.name for r in CrossoverTypes]),
    type=str,
)

ValidSelection = Enum(
    'ValidSelection',
    ((value, value) for value in [r.name for r in SelectionTypes]),
    type=str,
)

ValidMultiObjectiveSelection = Enum(
    'ValidMultiObjectiveSelection',
    ((value, value) for value in [r.name for r in MultiObjectiveSelectionTypes]),
    type=str,
)

ValidOptimizer = Enum(
    'ValidOptimizer',
    ((value, value) for value in [r.name for r in OptimizerTypes]),
    type=str,
)


class OptimizationParams(BaseModel):
    """GEFEST configuration dataclass.

    Attributes:
        crossover_each_prob (Optional[list[float]]): Probability for each crossover operation.
    """

    n_steps: int
    """Number optimization algorithm steps."""

    pop_size: int
    """Popultion size limit. Also auto-generated initial population size."""

    domain: Domain
    """Task domain."""

    objectives: list[Objective]
    """Task objectives."""

    mutations: Union[list[Callable[[Any], Structure]], list[ValidMutations]]
    """Mutation operations."""

    crossovers: Union[list[Callable[[Any], tuple[Structure]]], list[ValidCrossovers]]
    """Crossover operations."""

    selector: Union[Callable, ValidSelection]
    """Selection operation."""

    multiobjective_selector: Union[Callable, ValidMultiObjectiveSelection] = 'moead'
    """Evaluates one-dimensional fitness for multi objective.
    Uses `selector` to select individuals by the produces values.
    """
    moead_multi_objective_selector_neighbors: int = 2
    """Parameter used in moead selector."""

    optimizer: ValidOptimizer = 'gefest_ga'
    """Optimizer."""

    pair_selector: Callable = panmixis
    """Pairs selector for crossover operation."""

    sampler: Callable = StandardSampler
    """Structures sampler."""

    postprocessor: Callable = Postrocessor.apply_postprocess
    """Postprocessing module."""

    postprocess_rules: Union[list[Union[PolygonRule, StructureRule]], list[ValidRules]]
    """Postprocessing rules."""

    mutation_strategy: str = 'MutationStrategy'
    """Incapsulates mutation step logic."""

    crossover_strategy: str = 'CrossoverStrategy'
    """Incapsulates crossover step logic."""

    postprocess_attempts: int = 3
    """Nuber of attempts to correct invalid structures on postprocessing."""

    mutation_prob: float = 0.6
    """Probability to mutate structure."""

    crossover_prob: float = 0.6
    """Probability to crossover pair."""

    mutation_each_prob: Optional[list[float]] = None
    """Probability of each provided mutation operation to be applied."""

    crossover_each_prob: Optional[list[float]] = None
    """Probability of each provided crossover operation to be applied."""

    extra: int = 5
    """Number of extra structures to generate on each optimization step."""

    estimation_n_jobs: Optional[int] = 1
    """Number of cores for estimator parallel execution.
    Use more than 1 core only if you are sure that a particular estimator supports it!
    For example, if the estimator uses an external simulator,
    uses parallel calculations by itself,
    then increasing the number of jobs in the code may not have an effect
    or even slow down the execution of the program.
    """

    n_jobs: Optional[int] = -1
    """Nuber of cores to use in parallel execution of reproduction operations in GEFEST.
        n_jobs = -1 to use all cores,
        n_jobs > 0 to spicic number of cores,
        n_jobs = 0 to eval sequentially without joblib,
        use 0 for debug GEFEST only.
    """

    early_stopping_timeout: int = 60
    """GOLEM argument.
    For details see https://thegolem.readthedocs.io/en/latest/api/parameters.html
    """

    early_stopping_iterations: int = 1000
    """GOLEM argument.
    For details see:
        https://thegolem.readthedocs.io/en/latest/api/parameters.html
    """

    golem_adapter: Callable = StructureAdapter
    """Adapter for convertation GEFEST Structure into GOLEM OptGraph."""

    tuner_cfg: Optional[TunerParams] = None
    """Configuration for GOLEM tuner wrap."""

    log_dir: str = 'logs'
    """Directory for logs."""

    run_name: str = 'run_name'
    """Experiment name."""

    log_dispatcher: Callable = LogDispatcher
    """GEFEST logger."""

    golem_keep_histoy: bool = False
    """Enables/disables GOLEM experiment historry serialization."""

    golem_genetic_scheme_type: GeneticSchemeTypesEnum = 'steady_state'
    """GOLEM configuration argument `genetic_scheme_type`.
        For details see:
            https://thegolem.readthedocs.io/en/latest/api/parameters.html
    """

    golem_adaptive_mutation_type: MutationAgentTypeEnum = 'default'
    """GOLEM configuration argument `adaptive_mutation_type`.
        For details see:
            https://thegolem.readthedocs.io/en/latest/api/parameters.html
    """

    golem_selection_type: SelectionTypesEnum = 'spea2'
    """GOLEM configuration argument `selection_type`.
        For details see:
            https://thegolem.readthedocs.io/en/latest/api/parameters.html
    """

    golem_surrogate_each_n_gen: int = 5
    """Frequency of usage surrogate model in `golem_surrogate` optimizer"""

    @model_validator(mode='after')
    def create_classes_instances(self):
        """Selects and initializes specified modules."""
        if isinstance(self.optimizer, str):
            self.optimizer = getattr(OptimizerTypes, self.optimizer).value

        if isinstance(self.postprocess_rules[0], str):
            self.postprocess_rules = [getattr(Rules, name).value for name in self.postprocess_rules]

        if isinstance(self.mutations[0], str):
            self.mutations = [getattr(MutationTypes, name).value.func for name in self.mutations]

        if isinstance(self.crossovers[0], str):
            self.crossovers = [getattr(CrossoverTypes, name).value.func for name in self.crossovers]

        if isinstance(self.golem_genetic_scheme_type, str):
            self.selector = getattr(GeneticSchemeTypesEnum, self.golem_genetic_scheme_type.value)

        if isinstance(self.selector, str):
            self.selector = getattr(SelectionTypes, self.selector)

        if isinstance(self.multiobjective_selector, str):
            self.multiobjective_selector = getattr(
                MultiObjectiveSelectionTypes,
                self.multiobjective_selector,
            ).value

        if self.mutation_each_prob is None:
            self.mutation_each_prob = [1 / len(self.mutations) for _ in range(len(self.mutations))]

        if self.crossover_each_prob is None:
            self.crossover_each_prob = [
                1 / len(self.crossovers) for _ in range(len(self.crossovers))
            ]

        self.postprocessor = partial(
            self.postprocessor,
            domain=self.domain,
            rules=self.postprocess_rules,
            attempts=self.postprocess_attempts,
        )
        self.sampler = self.sampler(opt_params=self)
        self.golem_adapter = self.golem_adapter(self.domain)
        self.log_dispatcher = self.log_dispatcher(self.log_dir, self.run_name)

        return self

    model_config = ConfigDict({'arbitrary_types_allowed': True})
