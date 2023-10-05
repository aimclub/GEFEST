import pickle
from functools import partial

import numpy as np
from hyperopt import hp

from gefest.core.algs.postproc.resolve_errors import Rules, postprocess
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
from gefest.core.opt.tuning.tuner import GolemTuner
from gefest.core.structure.prohibited import create_prohibited
from gefest.core.viz.struct_vizualizer import GIFMaker
from gefest.tools.estimators.simulators.sound_wave.sound_interface import (
    SoundSimulator,
    generate_map,
)
from gefest.tools.fitness import Fitness
from gefest.tools.optimizers.GA.base_GA import BaseGA

# pre domain params
grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300  # Number of points on y-axis
coord_X = np.linspace(20, 100, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(20, 100, grid_resolution_y + 1)  # Y coordinate for spatial grid
grid = [grid_resolution_x, grid_resolution_y]  # points grid
fixed_area = [[[45, 55], [55, 55], [55, 45], [45, 45], [45, 55]]]


def load_file_from_path(path: str):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        f.close()
    return file


if __name__ == '__main__':

    #  in the future all model can be loaded from configs

    #  domain configuration
    geometry = Geometry2D(is_closed=True, is_convex=True)
    prohibited = create_prohibited(1, [], [], fixed_area=fixed_area)
    domain = Domain(
        allowed_area=[
            (min(coord_X), min(coord_Y)),
            (min(coord_X), max(coord_Y)),
            (max(coord_X), max(coord_Y)),
            (max(coord_X), min(coord_Y)),
            (min(coord_X), min(coord_Y)),
        ],
        geometry=geometry,
        max_poly_num=1,
        min_poly_num=1,
        max_points_num=30,
        min_points_num=20,
        prohibited_area=prohibited,
    )

    #  tuner config
    tp = TunerParams(
        tuner_type='iopt',
        n_steps_tune=2,
        sampling_variance=0.1,
        hyperopt_dist=hp.uniform,
    )

    #  fitness function
    class SoundFieldFitness(Fitness):
        def __init__(self, domain, estimator, path_best_struct=None):
            super().__init__(domain, SoundSimulator(domain=domain))
            self.path_best_struct = path_best_struct

            if self.path_best_struct is None:
                print('please, set up the best spl matrix into configuration')
                print('the best structure will be generated randomly')
                rnd_structure = get_random_structure(domain)
                best_spl = generate_map(domain, rnd_structure)
            else:
                best_structure = load_file_from_path(path_best_struct)
                best_spl = self.estimator(best_structure)
                best_spl = np.nan_to_num(best_spl, nan=0, neginf=0, posinf=0)

            self.best_spl = best_spl

        def fitness(self, ind: Structure):
            spl = self.estimator(ind)
            current_spl = np.nan_to_num(spl, nan=0, neginf=0, posinf=0)

            l_f = np.sum(np.abs(self.best_spl - current_spl))
            return l_f

    #  fitness estimator
    estimator = SoundFieldFitness(
        domain,
        SoundSimulator(domain, None),
        None,
    )

    #  optimization params config
    opt_params = OptimizationParams(
        crossovers=[
            partial(polygon_level_crossover, domain=domain),
            partial(structure_level_crossover, domain=domain),
        ],
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
        mutation_each_prob=[0.125, 0.125, 0.25, 0.25, 0.00, 0.00, 0.25],
        pair_selector=panmixis,
        postprocess_attempts=3,
        domain=domain,
        postprocessor=postprocess,
        estimator=estimator,
        postprocess_rules=[
            Rules.not_out_of_bounds.value,
            Rules.not_closed_polygon.value,
            Rules.not_self_intersects.value,
            Rules.not_too_close_polygons.value,
            Rules.not_overlaps_prohibited.value,
            Rules.not_too_close_points.value,
        ],
        extra=3,
        n_jobs=-1,
        golem_adapter=StructureAdapter,
        tuner_cfg=tp,
        n_steps=1,
        pop_size=3,
    )

    optimizer = BaseGA(opt_params)
    optimized_pop = optimizer.optimize()

    #  make mp4 of optimized pop here if need

    tuner = GolemTuner(opt_params)
    n_best_for_tune = 1
    tuned_individuals = tuner.tune(optimized_pop[0:n_best_for_tune])

    #  make mp4 of tuned pop here if need

    #  code to create mp4
    ###
    #  gm = GIFMaker(domain=domain)  # mp4 maker actually
    #  gm.create_frame(_structure_, {'Optimized': _structure_.fitness}) #  make frames for each stucture you want
    #  gm.make_gif('tuning', 500, ) #  save file
