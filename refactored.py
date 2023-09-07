from functools import partial
from typing import Callable, Optional

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from gefest.core.algs.geom.validation import *
from gefest.core.algs.postproc.resolve_errors import *
from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import *
from gefest.core.opt.operators.crossovers import panmixis, structure_level_crossover
from gefest.core.opt.operators.mutations import *
from gefest.core.opt.operators.selections import tournament_selection
from gefest.core.opt.strategies.crossover import CrossoverStrategy
from gefest.core.opt.strategies.mutation import MutationStrategy
from gefest.core.opt.strategies.strategy import Strategy
from gefest.core.structure.prohibited import create_prohibited
from gefest.core.utils.mp_manager import WorkersManager
from gefest.core.viz.struct_vizualizer import StructVizualizer
from gefest.tools.optimizers.optimizer import Optimizer
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
    sampler: Callable
    estimator: Callable
    postprocessor: Callable
    postprocess_rules: dict[str, list[Callable, Callable]]
    mutation_prob: float = 0.6
    crossover_prob: float = 0.6
    mutation_each_prob: Optional[list[float]] = None
    crossover_each_prob: Optional[list[float]] = None
    workers_manager: Optional[object] = None

    def __post_init__(self):
        self.crossovers = [
            partial(fun, domain=self.domain)
            for fun in self.crossovers
        ]
        self.postprocessor = partial(
            self.postprocessor,
            domain=self.domain,
            rule_fix_pairs=self.postprocess_rules,
        )
        self.sampler = self.sampler(opt_params=self)


@logger.catch
def main(opt_params):

    optimizer = BaseGA(opt_params)

    optimizer.optimize(1)

    from gefest.core.viz.struct_vizualizer import StructVizualizer

    plt.figure(figsize=(7, 7))
    visualiser = StructVizualizer(domain)

    info = {
        'spend_time': 1,
        'fitness': optimizer._pop[0].fitness,
        'type': 'prediction',
    }
    visualiser.plot_structure(
        [optimizer._pop[0]], [info], ['-'],
    )

    plt.show(block=True)

from contextlib import contextmanager


@contextmanager
def configuration(subprocess_holder):
    try:
        # load config from yaml
        yield subprocess_holder
    except Exception:
        raise
    finally:
        # add simulators safe exit
        subprocess_holder.workers_manager.pool.close()
        subprocess_holder.workers_manager.pool.terminate()


if __name__ == '__main__':
    logger.add('somefile.log', enqueue=True)
    class BaseGA(Optimizer):
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

        def optimize(self, n_steps: int) -> list[Structure]:
            for _ in tqdm(range(n_steps)):
                self._pop = self.estimator(pop=self._pop)
                self._pop = self.selector(self._pop, self.opt_params.pop_size)
                self._pop = self.crossover(self._pop)
                self._pop = self.mutation(self._pop)

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

    # domain = Domain(
    #     allowed_area=[
    #         (-125, 100),
    #         (-75, 170),
    #         (15, 170),
    #         (30, 90),
    #         (-20, -130),
    #         (-20, -170),
    #         (-125, -170),
    #         (-125, 100),
    #     ],
    #     geometry=Geometry2D(is_closed=True),
    #     max_poly_num=6,
    #     min_poly_num=1,
    #     max_points_num=16,
    #     min_points_num=5,
    #     is_closed=True,
    # )

    domain = Domain(
        allowed_area=[
            (0, 0),
            (0, 100),
            (100, 100),
            (100, 0),
            (0, 0),
        ],
        geometry=Geometry2D(is_closed=True),
        max_poly_num=5,
        min_poly_num=1,
        max_points_num=20,
        min_points_num=6,
        is_closed=True,
    )
    from pathlib import Path

    from gefest.core.geometry import PolyID
    from gefest.tools import Estimator
    from gefest.tools.estimators.simulators.swan.swan_interface import Swan

    # root_path = Path(__file__).parent.parent.parent.parent
    # path_sim = (
    #     'F:/Git_Repositories/gefest_fork/GEFEST/gefest/tools/estimators/simulators/swan/swan_model/'
    # )
    # from gefest.tools import Fitness
    # from gefest.tools.estimators.simulators.swan.swan_interface import Swan
    # class SwanFitness(Fitness):
    #     def __init__(self, estimator, domain, max_length):
    #         super().__init__(estimator)
    #         self.domain = domain
    #         self.max_length = max_length
    #     def fitness(self, ind: Structure):
    #         lengths = 0
    #         for poly in ind:
    #             if poly.id_ != PolyID.FIXED_POLY:
    #                 length = self.domain.geometry.get_length(poly)
    #                 lengths += length
    #         _, hs = self.estimator(struct=ind)
    #         l_f = lengths / (2 * self.max_length) + hs
    #         return l_f
    # class ComsolFitness(Fitness):
    #     def __init__(self, simulator, domain):
    #         super().__init__(simulator)
    #         self.domain = domain
    #     def fitness(self, ind: Structure):
    #         ind = self.estimator(ind)
    #         if ind is None:
    #             print("None")
    #         return ind
    # estimator = SwanFitness(
    #     swan,
    #     domain,
    #     np.linalg.norm(np.array([max(coord_X) - min(coord_X), max(coord_Y) - min(coord_Y)])),
    # ) 
    # from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol
    # comsol = Comsol('C:\\Users\\mangaboba\\Downloads\\rbc-trap-setup.mph')
    # estimator = ComsolFitness(simulator=comsol, domain=domain)

    opt_params = OptimizationParams(
        crossovers=[partial(structure_level_crossover, domain=domain)],
        mutations=[
            rotate_poly,
            resize_poly,
            add_point,
            drop_point,
            add_poly,
            drop_poly,
            pos_change_point_mutation,
        ],
        mutation_strategy=MutationStrategy,
        crossover_strategy=CrossoverStrategy,
        mutation_prob=0.6,
        crossover_prob=0.6,
        crossover_each_prob=[1],
        mutation_each_prob=[0.125, 0.125, 0.35, 0.05, 0.1, 0.05, 0.2],
        n_steps=5,
        pop_size=50,
        selector=tournament_selection,
        pair_selector=panmixis,
        postprocess_attempts=3,
        domain=domain,
        postprocessor=postprocess,
        postprocess_rules={
            'unclosed': [
                unclosed_poly,
                correct_unclosed_poly,
            ],
            'self_intersect': [
                self_intersection,
                correct_self_intersection,
            ],
            'wrong_points': [
                out_of_bound,
                correct_wrong_point,
            ],
        },
        sampler=StandardSampler,
        estimator=partial(area_length_ratio, domain=domain),
        workers_manager=WorkersManager(),
    )

    with configuration(opt_params):
        optimizer = BaseGA(opt_params)

        optimizer.optimize(100)

        from gefest.core.viz.struct_vizualizer import StructVizualizer
        plt.figure(figsize=(7, 7))
        visualiser = StructVizualizer(domain)

        info = {
            'spend_time': 1,
            'fitness': optimizer._pop[0].fitness,
            'type': 'prediction',
        }
        visualiser.plot_structure(
            [optimizer._pop[0]], [info], ['-'],
        )

        plt.show(block=True)
