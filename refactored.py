import copy
import math
from functools import partial
from random import randint
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

logger.add("somefile.log", enqueue=True)

from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass

from cases.synthetic.circle.configuration.circle_estimator import configurate_estimator
from gefest.core import Geometry2D
from gefest.core.algs.geom.validation import *
from gefest.core.algs.postproc.resolve_errors import *
from gefest.core.geometry import Structure
from gefest.core.geometry.utils import *
from gefest.core.opt.abstract.strategy import Strategy
from gefest.core.opt.domain import Domain
from gefest.core.opt.operators.crossover import panmixis, structure_level_crossover
from gefest.core.opt.operators.mutation import *
from gefest.core.opt.operators.selection import tournament_selection
from gefest.core.opt.strategies.crossover import CrossoverStrategy
from gefest.core.opt.strategies.mutation import MutationStrategy
from gefest.tools.samplers.standard.standard import StandardSampler


@dataclass
class OptimizationParams:
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
    postprocessor: Callable
    sampler: Callable
    estimator: Callable
    mutation_prob: float = 0.6
    crossover_prob: float = 0.6
    mutation_each_prob: Optional[list[float]] = None
    crossover_each_prob: Optional[list[float]] = None


class BaseGA:
    def __init__(
        self,
        opt_params: OptimizationParams,
        **kwargs,
    ):
        self.opt_params: OptimizationParams = opt_params
        self.crossover: Strategy = opt_params.crossover_strategy(opt_params)
        self.mutation: Strategy = opt_params.mutation_strategy(opt_params)
        self.sampler: Callable = opt_params.sampler
        self.estimator: Callable[[list[Structure]], list[Structure]] = opt_params.estimator
        self.selector: Callable = opt_params.selector
        self.pop_size = opt_params.pop_size
        self._pop: list[Structure] = self.sampler(self.opt_params.pop_size)
        self.domain = self.opt_params.domain

    def solution(self, n_steps: int):
        for i in tqdm(range(n_steps)):

            self._pop = self.estimator(pop=self._pop)
            self._pop = self.selector(self._pop, self.opt_params.pop_size)
            self._pop = self.crossover(self._pop)

            self._pop = self.mutation(self._pop)
            print(len(self._pop))

        self._pop = sorted(self._pop, key=lambda x: x.fitness)
        return self._pop


def area_length_ratio(pop: list[Structure], domain: Domain):
    for ind in pop:
        ind.fitness = domain.max_x * domain.max_y
        losses = []
        for idx, poly in enumerate(ind):
            area = domain.geometry.get_square(poly)
            length = domain.geometry.get_length(poly)

            if area == 0:
                losses.append(sum(losses[::idx]) / idx)
            else:
                losses.append(1 - 4 * np.pi * area / length ** 2)
        if len(losses) > 0:
            ind.fitness = sum(losses)
    return pop


if __name__ == "__main__":

    is_closed = True
    X_domain_max = 300
    X_domain_min = 0
    Y_domain_max = 300
    Y_domain_min = 0
    domain = Domain(
        name="bruh",
        allowed_area=[
            (X_domain_min, Y_domain_min),
            (X_domain_min, Y_domain_max),
            (X_domain_max, Y_domain_max),
            (X_domain_max, Y_domain_min),
            (X_domain_min, Y_domain_min),
        ],
        geometry=Geometry2D(),
        max_poly_num=5,
        min_poly_num=1,
        max_points_num=24,
        min_points_num=9,
        is_closed=True,
    )

    postprocessor = partial(
        postprocess,
        domain=domain,
        rule_fix_pairs={
            "unclosed": [
                partial(unclosed_poly, domain=domain),
                correct_unclosed_poly,
            ],
            "self_intersect": [
                self_intersection,
                partial(correct_self_intersection, domain=domain),
            ],
            "wrong_points": [
                partial(out_of_bound, domain=domain),
                partial(correct_wrong_point, domain=domain),
            ],
        },
    )
    opt_params = OptimizationParams(
        crossovers=[partial(structure_level_crossover, domain=domain)],
        mutations=[
            partial(rotate_poly, domain=domain),
            partial(resize_poly, domain=domain),
            partial(add_point, domain=domain),
            partial(drop_point, domain=domain),
            partial(add_poly, domain=domain),
            partial(drop_poly, domain=domain),
        ],
        mutation_strategy=MutationStrategy,
        crossover_strategy=CrossoverStrategy,
        mutation_prob=0.6,
        crossover_prob=0.6,
        crossover_each_prob=[1],
        mutation_each_prob=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
        n_steps=5,
        pop_size=50,
        selector=tournament_selection,
        pair_selector=panmixis,
        postprocess_attempts=3,
        domain=domain,
        postprocessor=postprocessor,
        sampler=StandardSampler(postprocessor=postprocessor, domain=domain),
        estimator=partial(area_length_ratio, domain=domain),
    )

    optimizer = BaseGA(opt_params)
    optimizer.solution(50)

    from gefest.core.viz.struct_vizualizer import StructVizualizer

    plt.figure(figsize=(7, 7))
    visualiser = StructVizualizer(domain)

    info = {
        "spend_time": 1,
        "fitness": optimizer._pop[0].fitness,
        "type": "prediction",
    }
    visualiser.plot_structure([optimizer._pop[0]], [info], ["-"])
    plt.show()
