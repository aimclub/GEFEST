import pickle
from functools import partial

from cases.sound_waves.poly_from_point import poly_from_comsol_txt
from core.utils.dice import dice_metric
from gefest.tools import Estimator
from gefest.core.algs.postproc.resolve_errors import Rules, apply_postprocess
import numpy as np
from hyperopt import hp
from gefest.core.opt.operators.selections import tournament_selection,roulette_selection,roulette_selection_sepa_2
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
import json
from gefest.core.opt.tuning.tuner import GolemTuner
from gefest.core.structure.prohibited import create_prohibited
from gefest.core.viz.struct_vizualizer import GIFMaker
from gefest.tools.estimators.simulators.sound_wave.sound_interface import (
    SoundSimulator,
    generate_map,
)
from gefest.tools.fitness import Fitness
from gefest.tools.optimizers.SPEA2.SPEA2 import SPEA2
from gefest.tools.optimizers.GA.base_GA import BaseGA
from pathlib import Path
from tools.estimators.simulators.swan.swan_interface import Swan
from shapely.geometry import shape
# pre domain params
root_path = Path(__file__).parent.parent.parent.parent

###########
grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300  # Number of points on y-axis
coord_X = np.linspace(20, 100, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(20, 100, grid_resolution_y + 1)  # Y coordinate for spatial grid
grid = [grid_resolution_x, grid_resolution_y]
fixed_area = None
#targets = [[i,11] for i in [10,12,14,16]]
#[14,11],[16,11],[18,11]

def load_file_from_path(path: str):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        f.close()
    return file


if __name__ == '__main__':
    #  domain configuration
    geometry = Geometry2D(is_closed=True, is_convex=False)
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
        min_points_num=7,
        polygon_side=0.00001

    )

    #  tuner config
    tp = TunerParams(
        tuner_type='sequential',
        n_steps_tune=25,
        sampling_variance=1,
        hyperopt_dist=hp.uniform,
    )
    path_=f"{root_path}/cases/synthetic/syn_gen/Comsol_points/lightning.txt"

    #Estimator
    class MetricEstimator(Estimator):
        def __init__(self,path_to_search_fig=''):
            self.best_structure = poly_from_comsol_txt(path=path_to_search_fig)
        def estimate(self, struct: Structure):
            dice_loss = 1-dice_metric(self.best_structure,struct)
            if dice_loss<0:
                return 2000
            return dice_loss

    syn_estimator = MetricEstimator(path_to_search_fig=path_)
    #  fitness function
    class SoundFieldFitness(Fitness):
        def __init__(self, domain, estimator):
            super().__init__(domain)
            self.estimator = estimator
        def fitness(self, ind: Structure):
            fitness = self.estimator(ind)
            return fitness

    #  fitness estimator
    estimator = SoundFieldFitness(
        domain,
        syn_estimator
    )

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
            pos_change_point_mutation,
        ],
        mutation_each_prob=[0.25, 0.25,0.5],
        pair_selector=panmixis,
        postprocess_attempts=30,
        domain=domain,
        postprocessor=apply_postprocess,
        estimator=estimator,
        postprocess_rules=[
            Rules.not_out_of_bounds.value,
            Rules.not_closed_polygon.value,
            Rules.not_self_intersects.value,
            Rules.not_too_close_polygons.value,
            Rules.not_overlaps_prohibited.value,
            Rules.not_too_close_points.value,
        ],
        extra=2000,
        n_jobs=-1,
        golem_adapter=StructureAdapter,
        tuner_cfg=tp,
        n_steps=100,
        pop_size=1000,
        selector=roulette_selection
    )
    optimizer = BaseGA(opt_params)

    optimized_pop = optimizer.optimize()

    #  make mp4 of optimized pop here if need

    # tuner = GolemTuner(opt_params)
    # n_best_for_tune = 1
    # tuned_individuals = tuner.tune(optimized_pop[0:n_best_for_tune])

    #  make mp4 of tuned pop here if need

    #  code to create mp4
    ###
    #  gm = GIFMaker(domain=domain)  # mp4 maker actually
    #  gm.create_frame(_structure_, {'Optimized': _structure_.fitness}) #  make frames for each stucture you want
    #  gm.make_gif('tuning', 500, ) #  save file
