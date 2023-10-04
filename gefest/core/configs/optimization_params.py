from functools import partial
from typing import Callable, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator

from gefest.core.algs.postproc.resolve_errors import PolygonRule, StructureRule
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.domain import Domain
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.tools.fitness import Fitness


class OptimizationParams(BaseModel):
    mutations: list[Callable]
    crossovers: list[Callable]
    mutation_strategy: Callable
    crossover_strategy: Callable
    n_steps: int
    pop_size: int
    postprocess_attempts: int
    domain: Domain
    selector: Callable
    pair_selector: Callable
    sampler: Callable
    estimator: Fitness
    postprocessor: Callable
    postprocess_rules: list[Union[PolygonRule, StructureRule]]
    mutation_prob: float = 0.6
    crossover_prob: float = 0.8
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
