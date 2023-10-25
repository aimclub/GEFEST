from functools import partial
from typing import Optional

import numpy as np
from hyperopt import hp
from sqlalchemy import lambda_stmt

from gefest.core.algs.postproc.resolve_errors import Postrocessor
from gefest.core.algs.postproc.rules import Rules
from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.geometry.utils import get_random_structure
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.core.opt.operators.crossovers import (
    panmixis,
    polygon_level_crossover,
    structure_level_crossover,
)
from gefest.core.opt.operators.mutations import (
    add_point,
    add_poly,
    drop_point,
    drop_poly,
    pos_change_point_mutation,
    resize_poly,
    rotate_poly,
)
from gefest.core.opt.operators.selections import (
    roulette_selection,
    tournament_selection,
)
from gefest.core.opt.tuning.tuner import GolemTuner
from gefest.core.structure.prohibited import create_prohibited
from gefest.core.viz.struct_vizualizer import GIFMaker
from gefest.tools import Estimator
from gefest.tools.objective import Objective
from gefest.tools.optimizers.GA.base_GA import BaseGA
from gefest.tools.optimizers.golem_optimizer.standard import StandardOptimizer

if __name__ == '__main__':

    #  domain configuration
    geometry = Geometry2D(is_closed=True, is_convex=True)
    # prohibited = create_prohibited(1, [], [], fixed_area=fixed_area)
    domain = Domain(
        allowed_area=[
            (0, 0),
            (0, 100),
            (100, 100),
            (100, 0),
            (0, 0),
        ],
        geometry=geometry,
        max_poly_num=1,
        min_poly_num=1,
        max_points_num=10,
        min_points_num=6,
    )

    tp = TunerParams(
        tuner_type='optuna',
        n_steps_tune=25,
        sampling_variance=1,
        hyperopt_dist=hp.uniform,
    )

    # metric 1
    class Area(Objective):
        def __init__(self, domain: Domain, estimator: Estimator = None) -> None:
            super().__init__(domain, estimator)

        def evaluate(self, ind: Structure) -> float:
            area = 0
            for poly in ind:
                area += self.domain.geometry.get_square(poly)
            area = abs(area - (50 * 50))
            norms = []
            points_num = 0
            if len(ind) == 1:
                for p1, p2 in zip(ind[0][:-1], ind[0][1:]):
                    norm = np.linalg.norm(np.array(p1.coords) - np.array(p2.coords))
                    norms.append(norm)
                points_num = len(ind[0])
            else:
                norms.append(1)
                points_num = sum(len(p) for p in ind)

            return area

    # metric 2
    class SideCoef(Objective):
        def __init__(self, domain: Domain, estimator: Estimator = None) -> None:
            super().__init__(domain, estimator)

        def evaluate(self, ind: Structure) -> float:
            area = 0
            for poly in ind:
                area += self.domain.geometry.get_square(poly)
            area = abs(area - (50 * 50))
            norms = []
            points_num = 0
            if len(ind) == 1:
                for p1, p2 in zip(ind[0][:-1], ind[0][1:]):
                    norm = np.linalg.norm(np.array(p1.coords) - np.array(p2.coords))
                    norms.append(norm)
                points_num = len(ind[0])
            else:
                norms.append(1)
                points_num = sum(len(p) for p in ind)

            sides_coef = points_num + min(norms) / max(norms)

            return sides_coef

    #  optimization params config
    opt_params = OptimizationParams(
        crossovers=[
            polygon_level_crossover,
            structure_level_crossover,
        ],
        crossover_prob=0.3,
        crossover_each_prob=[0.0, 1.0],
        mutations=[
            rotate_poly,
            resize_poly,
            add_point,
            drop_point,
            add_poly,
            drop_poly,
            pos_change_point_mutation,
        ],
        mutation_each_prob=[0.125, 0.125, 0.15, 0.35, 0.00, 0.00, 0.25],
        pair_selector=panmixis,
        postprocess_attempts=3,
        domain=domain,
        postprocessor=Postrocessor.apply_postprocess,
        objectives=[
            Area(domain=domain),
            SideCoef(domain=domain),
        ],
        postprocess_rules=[
            Rules.not_out_of_bounds.value,
            Rules.valid_polygon_geom.value,
            Rules.not_self_intersects.value,
            Rules.not_too_close_polygons.value,
            # Rules.not_overlaps_prohibited.value,
            Rules.not_too_close_points.value,
        ],
        tuner_cfg=tp,
        extra=0,
        n_jobs=1,
        golem_adapter=StructureAdapter,
        n_steps=1,
        pop_size=50,
    )

    optimizer = StandardOptimizer(opt_params)
    optimized_pop = optimizer.optimize()

    #  make mp4 of optimized pop here if need

    # pts = [[87.8183662653238, 23.51407028817777],
    #     [69.58377821212272, 34.2536199692584],
    #     [43.580509484692485, 92.58122224308758],
    #     [68.48530826424907, 94.10533753906533],
    #     [87.8183662653238, 23.51407028817777]] #899529.0143030069
    # s = Structure(polygons=[Polygon([Point(p[0], p[1]) for p in pts])], fitness=[899529.999])

    # opt_params.estimator.estimator.estimate(s)

    # gm = GIFMaker(domain=domain)  # mp4 maker actually
    # for st in optimizer.initial_pop:
    #     gm.create_frame(st, {'Optimized': st.fitness}) #  make frames for each stucture you want
    # gm.make_gif('diag_test', 500, ) #  save file

    from gefest.core.utils.functions import parse_structs

    # pop = parse_structs("C:\\Users\\mangaboba\\Downloads\\Telegram Desktop\\00075.log")
    # pop = sorted(pop, key=lambda x: x.fitness)
    # print(pop[0].fitness)
    tuner = GolemTuner(opt_params)
    n_best_for_tune = 1
    tuned_individuals = tuner.tune(opt_params.golem_adapter.restore(optimized_pop[0]))

    gm = GIFMaker(domain=domain)  # mp4 maker actually
    for st in optimized_pop:
        gm.create_frame(st, {'Optimized': st.fitness})  #  make frames for each stucture you want
    gm.make_gif(
        'diag_test',
        500,
    )  #  save file

    # tuned_individuals = tuner.tune(optimized_pop[0:n_best_for_tune])

    #  make mp4 of tuned pop here if need

    #  code to create mp4
    ###
    #  gm = GIFMaker(domain=domain)  # mp4 maker actually
    #  gm.create_frame(_structure_, {'Optimized': _structure_.fitness}) #  make frames for each stucture you want
    #  gm.make_gif('tuning', 500, ) #  save file
