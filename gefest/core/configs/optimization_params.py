from functools import partial
from typing import Callable, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator

from gefest.core.algs.postproc.resolve_errors import PolygonRule, StructureRule
from gefest.core.opt.strategies.crossover import CrossoverStrategy
from gefest.core.opt.strategies.mutation import MutationStrategy
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.domain import Domain
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.operators.selections import tournament_selection
from gefest.core.opt.operators.crossovers import panmixis
from gefest.tools.samplers.standard.standard import StandardSampler
from gefest.tools.fitness import Fitness


class OptimizationParams(BaseModel):
    mutations: list[Callable]
    crossovers: list[Callable]
    n_steps: int
    pop_size: int
    domain: Domain
    estimator: Fitness
    selector: Callable = tournament_selection
    pair_selector: Callable = panmixis
    sampler: Callable = StandardSampler
    postprocessor: Callable = postprocess
    postprocess_rules: list[Union[PolygonRule, StructureRule]]
    mutation_strategy: Callable = MutationStrategy
    crossover_strategy: Callable = CrossoverStrategy
    postprocess_attempts: int = 3
    mutation_prob: float = 0.6
    crossover_prob: float = 0.6
    mutation_each_prob: Optional[list[float]] = None
    crossover_each_prob: Optional[list[float]] = None
    n_jobs: Optional[int] = -1
    golem_adapter: Callable = StructureAdapter
    tuner_cfg: Optional[TunerParams] = None

    @model_validator(mode='after')
    def create_classes_instances(self):
        self.crossovers = [
            partial(fun, domain=self.domain) for fun in self.crossovers
        ]
        self.postprocessor = partial(
            self.postprocessor,
            domain=self.domain,
            rules=self.postprocess_rules,
        )
        self.sampler = self.sampler(opt_params=self)
        self.crossover_strategy = self.crossover_strategy(opt_params=self)
        self.mutation_strategy = self.mutation_strategy(opt_params=self)
        self.golem_adapter = self.golem_adapter(self.domain)

    model_config = ConfigDict({'arbitrary_types_allowed': True})
