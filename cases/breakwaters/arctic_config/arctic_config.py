import pickle
from pathlib import Path

import numpy as np
from breakwaters.breakwaters_utils import parse_arctic_geojson

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective
from gefest.tools.estimators.simulators.swan.swan_interface import Swan

root_path = Path(__file__).parent.parent.parent.parent
result_path = 'cases/breakwaters/newdata/result_PwiOsA2HE2igZUel.geojson'
border_path = 'cases/breakwaters/newdata/border_PwiOsA2HE2igZUel.geojson'


allow_water = parse_arctic_geojson(
    result_path=result_path,
    border_path=border_path,
    root_path=root_path
)
allow_area = [[74.80, 67.92], [74.80, 67.94]] + allow_water + [[74.80, 67.92]]
grid_resolution_x = 17  # Number of points on x-axis
grid_resolution_y = 31  # Number of points on y-axis
coord_y = np.linspace(
    min([p[1] for p in allow_area]) * 500,
    max([p[1] for p in allow_area]) * 500,
    grid_resolution_y + 1,
)  # X coordinate for spatial grid
coord_x = np.linspace(
    min([p[0] for p in allow_area]) * 500,
    max([p[0] for p in allow_area]) * 500,
    grid_resolution_x + 1,
)
grid = [grid_resolution_x, grid_resolution_y]  # points grid
fixed_area = None
targets = [[10, 15], [12, 14], [14, 14], [16, 14]]
# targets = [[i,11] for i in [10,12,14,16]]
# WINDWIND 19.1 225
# targets = [[14,10],[16,10],[18,10]]
# # # Metrics # # #


def load_file_from_path(path: str):
    """Func to load pickle file.

    :param path:
    :return:
    """
    with open(path, "rb") as f:
        _file = pickle.load(f)
        f.close()

    return _file


# # # Precompute domain arguments # # #

pass

# # #

domain_cfg = Domain(
    allowed_area=[
        (min(coord_x), min(coord_y)),
        (min(coord_x), max(coord_y)),
        (max(coord_x), max(coord_y)),
        (max(coord_x), min(coord_y)),
        (min(coord_x), min(coord_y)),
    ],
    name="main",
    min_poly_num=1,
    max_poly_num=4,
    min_points_num=3,
    max_points_num=4,
    polygon_side=0.0001,
    min_dist_from_boundary=0.0001,
    geometry_is_convex=False,
    geometry_is_closed=False,
    geometry="2D",
)

tuner_cfg = TunerParams(
    tuner_type="sequential",
    n_steps_tune=25,
    hyperopt_dist="uniform",
    verbose=True,
    timeout_minutes=60,
)
# # # Estimator # # #
path_ = f"{root_path}/cases/breakwaters/ob2_upd/"
swan_estimator = Swan(
    targets=targets,
    domain=domain_cfg,
    grid=grid,
    path=path_,
    hs_file_path="results/HSig_ob_example.dat",
)
# # # # # #


class BreakWatersFitness(Objective):
    """Class to init Objective for BreakWater case."""
    def __init__(self, domain, estimator):
        super().__init__(domain, estimator)
        self.estimator = estimator

    def _evaluate(self, ind: Structure):
        fitness = self.estimator(ind)
        return fitness


estimator = BreakWatersFitness(domain_cfg, swan_estimator)
opt_params = OptimizationParams(
    optimizer="gefest_ga",
    domain=domain_cfg,
    tuner_cfg=tuner_cfg,
    n_steps=4,
    pop_size=10,
    postprocess_attempts=15,
    mutation_prob=0.6,
    crossover_prob=0.6,
    mutations=[
        "rotate_poly",
        "resize_poly",
        "add_point",
        "drop_point",
        "add_poly",
        "drop_poly",
        "pos_change_point",
    ],
    selector="tournament_selection",
    mutation_each_prob=[0.125, 0.125, 0.15, 0.35, 0.00, 0.00, 0.25],
    crossovers=[
        "polygon_level",
        "structure_level",
    ],
    crossover_each_prob=[0.0, 1.0],
    postprocess_rules=[
        "not_out_of_bounds",
        "valid_polygon_geom",
        "not_self_intersects",
        "not_too_close_polygons",
        # 'not_overlaps_prohibited',
        "not_too_close_points",
    ],
    extra=4,
    n_jobs=-1,
    log_dir="logs",
    run_name="run_name",
    golem_keep_histoy=False,
    golem_genetic_scheme_type="steady_state",
    golem_surrogate_each_n_gen=5,
    objectives=[
        estimator,
    ],
)
