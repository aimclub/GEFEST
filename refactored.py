from functools import partial
from typing import Callable, Optional, Union

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from gefest.core.algs.postproc.resolve_errors import Rules, PolygonRule, StructureRule
from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import *
from gefest.core.opt.operators.crossovers import panmixis, structure_level_crossover, polygon_level_crossover
from gefest.core.opt.operators.mutations import *
from gefest.core.opt.operators.selections import tournament_selection, roulette_selection
from gefest.core.opt.strategies.crossover import CrossoverStrategy
from gefest.core.opt.strategies.mutation import MutationStrategy
from gefest.core.opt.strategies.strategy import Strategy
from gefest.core.structure.prohibited import create_prohibited
from gefest.tools.optimizers.optimizer import Optimizer
from gefest.tools.samplers.standard.standard import StandardSampler
from golem.core.optimisers.objective import ObjectiveEvaluate
from gefest.core.opt.adapters.structure import StructureAdapter
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.search_space import SearchSpace
from golem.core.optimisers.objective import Objective
from hyperopt import hp
from pydantic import RootModel

class LogDispatcher:
    def __init__(self) -> None:
        logger.warning("Logger configured outside.")

    def log_pop(self, pop: list[Structure], step: int):
        logger.log(3, f'Step {step}')
        for ind in pop:
            logger.log(4, RootModel[Structure](ind).model_dump())
        logger.info("Population logged. May be not sorted by fitness.")

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
    postprocess_rules: list[Union[PolygonRule, StructureRule]]
    mutation_prob: float = 0.6
    crossover_prob: float = 0.8
    mutation_each_prob: Optional[list[float]] = None
    crossover_each_prob: Optional[list[float]] = None
    n_jobs: Optional[int] = -1

    def __post_init__(self):
        self.crossovers = [
            partial(fun, domain=self.domain)
            for fun in self.crossovers
        ]
        self.postprocessor = partial(
            self.postprocessor,
            domain=self.domain,
            rules=self.postprocess_rules,
        )
        self.sampler = self.sampler(opt_params=self)
        self.crossover_strategy = self.crossover_strategy(opt_params=self)
        self.mutation_strategy = self.mutation_strategy(opt_params=self)


def search_space_generator(graph, pad):
    return {node.name : {
            param : { 
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [val - pad*3, val + pad*3],
                'type': 'continuous',
            } for param, val in zip(
                                    list(graph.nodes[idx_].content['params'].keys()),
                                    list(graph.nodes[idx_].content['params'].values()),
                                )
        } for idx_, node in enumerate(graph.nodes)}

if __name__ == '__main__':
    # logger.add('somefile.log', enqueue=True)

    class BaseGA(Optimizer):
        def __init__(
            self,
            opt_params: OptimizationParams,
            **kwargs,
        ):
            super().__init__(LogDispatcher())
            self.opt_params: OptimizationParams = opt_params
            self.crossover: Strategy = opt_params.crossover_strategy
            self.mutation: Strategy = opt_params.mutation_strategy
            self.sampler: Callable = opt_params.sampler
            self.estimator: Callable[[list[Structure]], list[Structure]] = opt_params.estimator
            self.selector: Callable = opt_params.selector
            self.pop_size = opt_params.pop_size
            self.n_steps = opt_params.n_steps
            self.domain = self.opt_params.domain
            self._pop: list[Structure] = self.sampler(self.opt_params.pop_size)
            self._pop = self.estimator(self._pop)
            self.logger.log_pop(self._pop, 'init')

        def optimize(self) -> list[Structure]:
            for step in tqdm(range(self.n_steps)):
                self._pop = self.selector(self._pop, self.opt_params.pop_size)
                self._pop = self.crossover(self._pop)
                self._pop = self.mutation(self._pop)
                self._pop.extend(self.sampler(5))
                self._pop = self.estimator(self._pop)
                self.logger.log_pop(self._pop, step)

            self._pop = sorted(self._pop, key=lambda x: x.fitness)
            [print(x.fitness) for x in self._pop]
            return self._pop


    def area_length_ratio(pop: list[Structure], domain: Domain):
        if isinstance(pop, Structure):
            pop = [pop]
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

    # domain = Domain( colsol
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

    

    # domain = Domain(
    #     allowed_area=[
    #         (0, 0),
    #         (0, 100),
    #         (100, 100),
    #         (100, 0),
    #         (0, 0),
    #     ],
    #     geometry=Geometry2D(is_closed=True, is_convex=True),
    #     max_poly_num=4,
    #     min_poly_num=2,
    #     max_points_num=8,
    #     min_points_num=4,
    #     polygon_side=0.01,
    # )
    # print(domain.dist_between_points)
    # print(domain.dist_between_polygons)
    # print(domain.min_dist_from_boundary)
    from pathlib import Path

    from gefest.core.geometry import PolyID
    from gefest.tools import Estimator
    from gefest.tools.estimators.simulators.swan.swan_interface import Swan


    grid_resolution_x = 83  # Number of points on x-axis
    grid_resolution_y = 58  # Number of points on y-axis
    coord_X = np.linspace(0, 2075, grid_resolution_x + 1)  # X coordinate for spatial grid
    coord_Y = np.linspace(0, 1450, grid_resolution_y + 1)  # Y coordinate for spatial grid

    grid = [grid_resolution_x, grid_resolution_y]  # points grid
    targets = [[49, 26], [11, 37], [5, 60]]  # grid coordinates of considered targets

    """
    Prohibited objects
    """
    fixed_area = [
        [[471, 5], [1335, 2], [1323, 214], [1361, 277], [1395, 327], [1459, 405], [1485, 490], [1449, 521], [1419, 558],
        [1375, 564], [1321, 469], [1248, 318], [1068, 272], [921, 225], [804, 231], [732, 266], [634, 331], [548, 405],
        [485, 482], [424, 569], [381, 625], [310, 662], [271, 684], [244, 706], [203, 708], [182, 647], [214, 638],
        [234, 632], [275, 588], [346, 475], [427, 366], [504, 240], [574, 166], [471, 5]],
        [[652, 1451], [580, 1335], [544, 1253], [468, 1190], [439, 1170], [395, 1150], [378, 1115], [438, 1070],
        [481, 1059], [508, 1076], [539, 1133], [554, 1183], [571, 1244], [594, 1305], [631, 1366], [657, 1414],
        [671, 1449], [652, 1451]]
    ]
    fixed_targets = [[coord_X[26], coord_Y[49]], [coord_X[37], coord_Y[11]], [coord_X[60], coord_Y[5]]]
    fixed_poly = [
        [[878, 1433], [829, 1303], [739, 1116], [619, 995], [447, 962], [306, 1004], [254, 1092], [241, 1184],
        [269, 1244],
        [291, 1338], [370, 1450]],
        [[878, 1433], [829, 1303], [739, 1116], [619, 995], [447, 962], [274, 868], [180, 813], [126, 717], [146, 580],
        [203, 480], [249, 469], [347, 471]]
    ]

    # Creation of prohibited structure consist of targets, lines, areas
    prohibited_structure = create_prohibited(
        targets=fixed_targets,
        fixed_area=fixed_area,
        fixed_points=fixed_poly,
    )

    domain = Domain(allowed_area=[(min(coord_X), min(coord_Y)),
                                  (min(coord_X), max(coord_Y)),
                                  (max(coord_X), max(coord_Y)),
                                  (max(coord_X), min(coord_Y)),
                                  (min(coord_X), min(coord_Y))],
                    prohibited_area=prohibited_structure,
                    geometry=Geometry2D(is_closed=False, is_convex=True),
                    max_poly_num=4,
                    min_poly_num=2,
                    max_points_num=16,
                    min_points_num=4,
                    polygon_side=0.01,
                    )

    # root_path = Path(__file__).parent.parent.parent.parent
    path_sim = (
        'F:/Git_Repositories/gef_ref/GEFEST/gefest/tools/estimators/simulators/swan/swan_model/'
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
                length = self.domain.geometry.get_length(poly)
                lengths += length
            # c.polygons.extend([
            #     p for p in domain.prohibited_area.polygons
            #     if p.id_ not in [PolyID.PROH_TARG]])
            idk, hs = self.estimator(struct=ind)
            l_f = [hs, 2 * lengths / self.max_length]  # lengths / (2 * self.max_length) + hs

            return l_f

    # class ComsolFitness(Fitness):
    #     def __init__(self, simulator, domain):
    #         super().__init__(simulator)
    #         self.domain = domain
    #     def fitness(self, ind: Structure):
    #         ind = self.estimator(ind)
    #         if ind is None:
    #             print("None")
    #         return ind
    estimator = SwanFitness(
        Swan(path=path_sim,
                targets=targets,
                grid=grid,
                domain=domain),
        domain,
        np.linalg.norm(np.array([max(coord_X) - min(coord_X), max(coord_Y) - min(coord_Y)])),
    )

    # from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol
    # comsol = Comsol('C:\\Users\\mangaboba\\Downloads\\rbc-trap-setup.mph')
    # estimator = ComsolFitness(simulator=comsol, domain=domain)


    opt_params = OptimizationParams(
        crossovers=[
            partial(polygon_level_crossover, domain=domain),
            partial(structure_level_crossover, domain=domain)],
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
        crossover_each_prob=[0.0, 1.0],
        mutation_each_prob=[0.125, 0.125, 0.25, 0.11, 0.07, 0.07, 0.25],
        n_steps=8,
        pop_size=5,
        selector=tournament_selection,
        pair_selector=panmixis,
        postprocess_attempts=3,
        domain=domain,
        postprocessor=postprocess,
        postprocess_rules=[

            Rules.not_out_of_bounds.value,
            Rules.not_closed_polygon.value,
            Rules.not_self_intersects.value,
            Rules.not_too_close_polygons.value,
            Rules.not_overlaps_prohibited.value,
            Rules.not_too_close_points.value,
        ],
        sampler=StandardSampler,
        estimator=estimator,
        n_jobs=-1,
    )

    # with configuration(opt_params):
    # optimizer = BaseGA(opt_params)
    from gefest.tools.optimizers.SPEA2.SPEA2 import SPEA2
    optimizer = SPEA2(opt_params)
    import cProfile
    logger.add(
                "individuals.log",
                level=3,
                format="{message}",
                filter=lambda record: record["level"].name in ['Level 3', 'Level 4'],
            )

    optimizer.optimize()



    # def tune_fintess(pop: Structure, domain: Domain):



    #     fitness = domain.max_x * domain.max_y
    #     losses = []
    #     for idx, poly in enumerate(pop):
    #         area = domain.geometry.get_square(poly)
    #         length = domain.geometry.get_length(poly)

    #         if area == 0:
    #             losses.append(sum(losses[::idx]) / idx)
    #         else:
    #             losses.append(1 - 4 * np.pi * area / length ** 2)
    #     if len(losses) > 0:
    #         fitness = sum(losses)
    #     return fitness

    # struct = copy.deepcopy(optimizer._pop[0])
    # objective = Objective(
    #     quality_metrics={
    #         'som_metric': partial(tune_fintess, domain=domain),
    #     },
    # )
    # adapted = StructureAdapter().adapt(struct)
    # objective_evaluator = ObjectiveEvaluate(objective=objective, eval_n_jobs=6)
    # tuner = IOptTuner(
    #     objective_evaluate=objective_evaluator,
    #     search_space=SearchSpace(search_space_generator(adapted, 1)),
    #     adapter=StructureAdapter(),
    #     iterations=500,
    # )
    # #  print(search_space_generator(struct, 0.05))
    # print(struct)

    # tuned = tuner.tune(adapted)
    # tuned = optimizer.estimator(tuned)[0]

    from gefest.core.viz.struct_vizualizer import GIFMaker
    # gm = GIFMaker(domain=domain)
    # gm.create_frame(optimizer._pop[0], {'Optimized': optimizer._pop[0].fitness})
    # gm.create_frame(tuned, {'Tuned': tuned.fitness})
    # gm.make_gif('tuning', 500, )

    gm = GIFMaker(domain=domain)
    for s in optimizer._pop:
        gm.create_frame(s, {'Fitness': s.fitness}, domain, )
    gm.make_gif('test', 500, )
