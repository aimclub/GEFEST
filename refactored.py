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
from gefest.core.geometry.utils import *
from gefest.core.opt.strategies.strategy import Strategy
from gefest.core.opt.domain import Domain
from gefest.core.opt.operators.crossover import panmixis, structure_level_crossover
from gefest.core.opt.operators.mutation import *
from gefest.core.opt.operators.selection import tournament_selection
from gefest.core.opt.strategies.crossover import CrossoverStrategy
from gefest.core.opt.strategies.mutation import MutationStrategy
from gefest.tools.samplers.standard.standard import StandardSampler


@logger.catch
def main():
    logger.add('somefile.log', enqueue=True)

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

        def __post_init__(self):
            self.mutations = [
                partial(fun, domain=self.domain)
                for fun in self.mutations
            ]
            self.crossovers = [
                partial(fun, domain=self.domain)
                for fun in self.crossovers
            ]
            self.postprocessor = partial(
                self.postprocessor,
                domain=domain,
                rule_fix_pairs=self.postprocess_rules,
            )
            self.sampler = self.sampler(opt_params=self)



    from gefest.tools.optimizers.optimizer import Optimizer
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

        def solution(self, n_steps: int) -> list[Structure]:
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




    # grid_resolution_x = 83  # Number of points on x-axis
    # grid_resolution_y = 58  # Number of points on y-axis
    # coord_X = np.linspace(0, 2075, grid_resolution_x + 1)  # X coordinate for spatial grid
    # coord_Y = np.linspace(0, 1450, grid_resolution_y + 1)  # Y coordinate for spatial grid

    # grid = [grid_resolution_x, grid_resolution_y]  # points grid
    # targets = [[49, 26], [11, 37], [5, 60]]  # grid coordinates of considered targets

    # """
    # Prohibited objects
    # """
    # fixed_area = [
    #     [
    #         [471, 5],
    #         [1335, 2],
    #         [1323, 214],
    #         [1361, 277],
    #         [1395, 327],
    #         [1459, 405],
    #         [1485, 490],
    #         [1449, 521],
    #         [1419, 558],
    #         [1375, 564],
    #         [1321, 469],
    #         [1248, 318],
    #         [1068, 272],
    #         [921, 225],
    #         [804, 231],
    #         [732, 266],
    #         [634, 331],
    #         [548, 405],
    #         [485, 482],
    #         [424, 569],
    #         [381, 625],
    #         [310, 662],
    #         [271, 684],
    #         [244, 706],
    #         [203, 708],
    #         [182, 647],
    #         [214, 638],
    #         [234, 632],
    #         [275, 588],
    #         [346, 475],
    #         [427, 366],
    #         [504, 240],
    #         [574, 166],
    #         [471, 5],
    #     ],
    #     [
    #         [652, 1451],
    #         [580, 1335],
    #         [544, 1253],
    #         [468, 1190],
    #         [439, 1170],
    #         [395, 1150],
    #         [378, 1115],
    #         [438, 1070],
    #         [481, 1059],
    #         [508, 1076],
    #         [539, 1133],
    #         [554, 1183],
    #         [571, 1244],
    #         [594, 1305],
    #         [631, 1366],
    #         [657, 1414],
    #         [671, 1449],
    #         [652, 1451],
    #     ],
    # ]
    # fixed_targets = [
    #     [coord_X[26], coord_Y[49]],
    #     [coord_X[37], coord_Y[11]],
    #     [coord_X[60], coord_Y[5]],
    # ]
    # fixed_poly = [
    #     [
    #         [878, 1433],
    #         [829, 1303],
    #         [739, 1116],
    #         [619, 995],
    #         [447, 962],
    #         [306, 1004],
    #         [254, 1092],
    #         [241, 1184],
    #         [269, 1244],
    #         [291, 1338],
    #         [370, 1450],
    #     ],
    #     [
    #         [878, 1433],
    #         [829, 1303],
    #         [739, 1116],
    #         [619, 995],
    #         [447, 962],
    #         [274, 868],
    #         [180, 813],
    #         [126, 717],
    #         [146, 580],
    #         [203, 480],
    #         [249, 469],
    #         [347, 471],
    #     ],
    # ]
    from gefest.core.structure.prohibited import create_prohibited

    # Creation of prohibited structure consist of targets, lines, areas
    # prohibited_structure = create_prohibited(
    #     targets=fixed_targets,
    #     fixed_area=fixed_area,
    #     fixed_points=fixed_poly,
    # )
    domain = Domain(
        allowed_area=[
            (-125, 100),
            (-75, 170),
            (15, 170),
            (30, 90),
            (-20, -130),
            (-20, -170),
            (-125, -170),
            (-125, 100),
        ],
        geometry=Geometry2D(is_closed=True),
        max_poly_num=5,
        min_poly_num=1,
        max_points_num=16,
        min_points_num=5,
        is_closed=True,
    )
    # domain = Domain(
    #     allowed_area=[
    #         (min(coord_X), min(coord_Y)),
    #         (min(coord_X), max(coord_Y)),
    #         (max(coord_X), max(coord_Y)),
    #         (max(coord_X), min(coord_Y)),
    #         (min(coord_X), min(coord_Y)),
    #     ],
    #     prohibited_area=prohibited_structure,
    #     max_poly_num=5,
    #     min_poly_num=3,
    #     max_points_num=10,
    #     min_points_num=6,
    #     geometry=Geometry2D(is_closed=False),
    # )

    # postprocessor = partial(
    #     postprocess,
    #     domain=domain,
    #     rule_fix_pairs={
    #         'unclosed': [
    #             unclosed_poly,
    #             correct_unclosed_poly,
    #         ],
    #         'self_intersect': [
    #             self_intersection,
    #             correct_self_intersection,
    #         ],
    #         'wrong_points': [
    #             out_of_bound,
    #             correct_wrong_point,
    #         ],
    #     },
    # )

    from gefest.tools import Estimator
    from gefest.tools.estimators.simulators.swan.swan_interface import Swan
    from pathlib import Path
    from gefest.core.geometry import PolyID

    root_path = Path(__file__).parent.parent.parent.parent
    path_sim = (
        'F:/Git_Repositories/gefest_fork/GEFEST/gefest/tools/estimators/simulators/swan/swan_model/'
    )

    from gefest.tools import Fitness
    from gefest.tools.estimators.simulators.swan.swan_interface import Swan

    class SwanFitness(Fitness):
        def __init__(self, estimator, domain, max_length):
            super().__init__(estimator)
            self.domain = domain
            self.max_length = max_length

        def fitness(self, ind: Structure):

            lengths = 0
            for poly in ind:
                if poly.id_ != PolyID.FIXED_POLY:
                    length = self.domain.geometry.get_length(poly)
                    lengths += length

            _, hs = self.estimator(struct=ind)
            l_f = lengths / (2 * self.max_length) + hs

            return l_f

    class ComsolFitness(Fitness):
        def __init__(self, simulator, domain):
            super().__init__(simulator)
            self.domain = domain

        def fitness(self, ind: Structure):
            ind = self.estimator(ind)
            if ind is None:
                print("None")
            return ind

    # estimator = SwanFitness(
    #     swan,
    #     domain,
    #     np.linalg.norm(np.array([max(coord_X) - min(coord_X), max(coord_Y) - min(coord_Y)])),
    # ) 

    from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol
    comsol = Comsol('C:\\Users\\mangaboba\\Downloads\\rbc-trap-setup.mph')
    estimator = ComsolFitness(simulator=comsol, domain=domain)

    opt_params = OptimizationParams(
        crossovers=[partial(structure_level_crossover, domain=domain)],
        mutations=[
            rotate_poly,
            resize_poly,
            add_point,
            drop_point,
            add_poly,
            drop_poly,
        ],
        mutation_strategy=MutationStrategy,
        crossover_strategy=CrossoverStrategy,
        mutation_prob=0.6,
        crossover_prob=0.6,
        crossover_each_prob=[1],
        mutation_each_prob=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
        n_steps=5,
        pop_size=3,
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
        estimator=estimator,
    )

    optimizer = BaseGA(opt_params)
    
    # try:
    optimizer.solution(1)
    # except Exception as e:
    #     print(e)
    # finally:
    #     del optimizer.mutation
        # del optimizer.crossover
    from gefest.core.viz.struct_vizualizer import StructVizualizer

    plt.figure(figsize=(7, 7))
    visualiser = StructVizualizer(domain)

    info = {
        'spend_time': 1,
        'fitness': optimizer._pop[0].fitness,
        'type': 'prediction',
    }
    visualiser.plot_structure(
        [optimizer._pop[0]], [info], ['-']
    )

    plt.show(block=True)

from gefest.core.utils.mp_manager import WorkersManager
if __name__ == '__main__':

    with WorkersManager():
        main()
