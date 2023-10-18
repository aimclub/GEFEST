import pickle
from functools import partial

import numpy as np
from hyperopt import hp
from gefest.core.opt.operators.selections import tournament_selection,roulette_selection
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
from gefest.tools.optimizers.GA.base_GA import BaseGA
from pathlib import Path
from tools.estimators.simulators.swan.swan_interface import Swan
from shapely.geometry import shape
# pre domain params
root_path = Path(__file__).parent.parent.parent
with open(f'{root_path}/cases/breakwaters/newdata/result_PwiOsA2HE2igZUel.geojson', 'r') as file:
    res_list = json.load(file)
with open(f'{root_path}/cases/breakwaters/newdata/border_PwiOsA2HE2igZUel.geojson', 'r') as file:
    border_dict = json.load(file)
border = shape(border_dict['features'][0]['geometry'])
water = [i for i in res_list['features'] if i['properties']['type'] =='water']
water_coord =[p['geometry']['coordinates'] for p in water]
cargo_piers = [i for i in res_list['features'] if i['properties']['type'] =='cargo_pier']
passenger_pier = [i for i in res_list['features'] if i['properties']['type'] =='passenger_pier']
piers = [i for i in res_list['features'] if (i['properties']['type']=='passenger_pier') or (i['properties']['type'] =='cargo_pier')]
piers_coords = [x[0] for x in [i['geometry']['coordinates'] for i in piers]]
piers_line = [max(p,key=lambda i: i[1]) for p in piers_coords]
unique_types = np.unique([i['properties']['type'] for i in res_list['features']])
allow_water = [i for i in water_coord[0][0] if (i[0]>74.8) and (i[1]<67.942) and (i[1]>67.915)]
###########
allow_area = [[74.80,67.92],[74.80,67.94]]+allow_water +[[74.80,67.92]]
grid_resolution_x = 17  # Number of points on x-axis
grid_resolution_y = 31  # Number of points on y-axis
coord_Y = np.linspace(min([p[1] for p in allow_area])*500, max([p[1] for p in allow_area])*500, grid_resolution_y + 1)  # X coordinate for spatial grid
coord_X = np.linspace(min([p[0] for p in allow_area])*500, max([p[0] for p in allow_area])*500, grid_resolution_x + 1)
grid = [grid_resolution_x, grid_resolution_y]  # points grid
fixed_area = None
targets = [[10,14],[10,16],[10,18]]


def load_file_from_path(path: str):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        f.close()
    return file


if __name__ == '__main__':
    # class SoundSimulator_(SoundSimulator):
    #     def __init__(self, domain, obstacle_map=None):
    #         super().__init__(domain, obstacle_map=None)
    #         self.duration = 200
    #         self.pressure_hist = np.zeros((self.duration, self.size_y, self.size_x))
    #         if (
    #                 obstacle_map is not None
    #                 and (obstacle_map.shape[0], obstacle_map.shape[1]) == self.map_size
    #         ):
    #             print("** Map Accepted **")
    #             self.obstacle_map = obstacle_map
    #         elif obstacle_map is not None and obstacle_map.shape != self.map_size:
    #             print("** Map size denied **")
    #             self.obstacle_map = np.zeros((self.size_y, self.size_x))
    #         else:
    #             self.obstacle_map = np.zeros((self.size_y, self.size_x))

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
        max_points_num=10,
        min_points_num=6,
        prohibited_area=prohibited,
    )

    #  tuner config
    tp = TunerParams(
        tuner_type='sequential',
        n_steps_tune=25,
        sampling_variance=1,
        hyperopt_dist=hp.uniform,
    )
    path_=f"{root_path}/cases/breakwaters/ob2"
    #Estimator
    swan_estimator = Swan(
        targets=targets,
        domain=domain,
        grid=grid,
        path=path_,
        hs_file_path='/results/HSig_ob_example.dat'
    )
    #  fitness function
    class SoundFieldFitness(Fitness):
        def __init__(self, domain, estimator):
            super().__init__(domain)
            self.estimator=estimator
        def fitness(self, ind: Structure):
            fitness = self.estimator(ind)
            return fitness

    #  fitness estimator
    estimator = SoundFieldFitness(
        domain,
        swan_estimator
    )

    #  optimization params config
    opt_params = OptimizationParams(
        crossovers=[
            partial(polygon_level_crossover, domain=domain),
            partial(structure_level_crossover, domain=domain),
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
        extra=2,
        n_jobs=-1,
        golem_adapter=StructureAdapter,
        tuner_cfg=tp,
        n_steps=2,
        pop_size=2,
        selector=roulette_selection
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
