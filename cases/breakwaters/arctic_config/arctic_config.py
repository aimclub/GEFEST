import json
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective
from gefest.tools.estimators.simulators.swan.swan_interface import Swan

root_path = Path(__file__).parent.parent.parent.parent
with open(
    f"{root_path}/cases/breakwaters/newdata/result_PwiOsA2HE2igZUel.geojson", "r"
) as file:
    res_list = json.load(file)
with open(
    f"{root_path}/cases/breakwaters/newdata/border_PwiOsA2HE2igZUel.geojson", "r"
) as file:
    border_dict = json.load(file)


border = shape(border_dict["features"][0]["geometry"])
water = [i for i in res_list["features"] if i["properties"]["type"] == "water"]
water_coord = [p["geometry"]["coordinates"] for p in water]
cargo_piers = [
    i for i in res_list["features"] if i["properties"]["type"] == "cargo_pier"
]
passenger_pier = [
    i for i in res_list["features"] if i["properties"]["type"] == "passenger_pier"
]
piers = [
    i
    for i in res_list["features"]
    if (i["properties"]["type"] == "passenger_pier")
    or (i["properties"]["type"] == "cargo_pier")
]
piers_coords = [x[0] for x in [i["geometry"]["coordinates"] for i in piers]]
piers_line = [max(p, key=lambda i: i[1]) for p in piers_coords]
unique_types = np.unique([i["properties"]["type"] for i in res_list["features"]])
allow_water = [
    i
    for i in water_coord[0][0]
    if (i[0] > 74.8) and (i[1] < 67.942) and (i[1] > 67.915)
]
allow_area = [[74.80, 67.92], [74.80, 67.94]] + allow_water + [[74.80, 67.92]]
grid_resolution_x = 17  # Number of points on x-axis
grid_resolution_y = 31  # Number of points on y-axis
coord_Y = np.linspace(
    min([p[1] for p in allow_area]) * 500,
    max([p[1] for p in allow_area]) * 500,
    grid_resolution_y + 1,
)  # X coordinate for spatial grid
coord_X = np.linspace(
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
    with open(path, "rb") as f:
        _file = pickle.load(f)
        f.close()
    return _file


# # # Precompute domain arguments # # #

pass

# # #

domain_cfg = Domain(
    allowed_area=[
        (min(coord_X), min(coord_Y)),
        (min(coord_X), max(coord_Y)),
        (max(coord_X), max(coord_Y)),
        (max(coord_X), min(coord_Y)),
        (min(coord_X), min(coord_Y)),
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
    def __init__(self, domain, estimator):
        super().__init__(domain, estimator)
        self.estimator = estimator

    def _evaluate(self, ind: Structure):
        fitness = self.estimator(ind)
        return fitness


#  fitness estimator
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
