from enum import Enum
from functools import partial
from typing import Callable, Optional, Union

from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from pydantic import BaseModel, ConfigDict, model_validator

from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.domain import Domain
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.core.opt.objective.objective import Objective
from gefest.core.opt.operators.crossovers import CrossoverTypes, panmixis
from gefest.core.opt.operators.mutations import MutationTypes
from gefest.core.opt.operators.selections import SelectionTypes
from gefest.core.opt.postproc.resolve_errors import Postrocessor
from gefest.core.opt.postproc.rules import PolygonRule, Rules, StructureRule
from gefest.core.utils.logger import LogDispatcher
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


class OptimizationParams(BaseModel):
    n_steps: int
    pop_size: int
    domain: Domain
    objectives: list[Objective]
    mutations: Union[list[Callable], list[ValidMutations]]
    crossovers: Union[list[Callable], list[ValidCrossovers]]
    selector: Union[Callable, ValidSelection]
    pair_selector: Callable = panmixis
    sampler: Callable = StandardSampler
    postprocessor: Callable = Postrocessor.apply_postprocess
    postprocess_rules: Union[list[Union[PolygonRule, StructureRule]], list[ValidRules]]
    mutation_strategy: str = 'MutationStrategy'
    crossover_strategy: str = 'CrossoverStrategy'
    postprocess_attempts: int = 3
    mutation_prob: float = 0.6
    crossover_prob: float = 0.6
    mutation_each_prob: Optional[list[float]] = None
    crossover_each_prob: Optional[list[float]] = None
    extra: int = 5
    n_jobs: Optional[int] = -1
    early_stopping_timeout: int = 60
    early_stopping_iterations: int = 1000
    golem_adapter: Callable = StructureAdapter
    tuner_cfg: Optional[TunerParams] = None
    log_dir: str = 'logs'
    run_name: str = 'run_name'
    log_dispatcher: Callable = LogDispatcher
    golem_keep_histoy: bool = False
    golem_genetic_scheme_type: GeneticSchemeTypesEnum = 'steady_state'
    golem_adaptive_mutation_type: MutationAgentTypeEnum = 'default'
    golem_selection_type: SelectionTypesEnum = 'spea2'
    golem_surrogate_each_n_gen: int = 5

    @model_validator(mode='after')
    def create_classes_instances(self):
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
