import numpy as np

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective
from gefest.tools.estimators.estimator import Estimator


# # # Metrics # # #
class Area(Objective):
    def __init__(self, domain: Domain, estimator: Estimator = None) -> None:
        super().__init__(domain, estimator)

    def evaluate(self, ind: Structure) -> float:
        area = 0
        for poly in ind:
            area += self.domain.geometry.get_square(poly)
        area = abs(area - (50 * 50))
        norms = []
        if len(ind) == 1:
            for p1, p2 in zip(ind[0][:-1], ind[0][1:]):
                norm = np.linalg.norm(np.array(p1.coords) - np.array(p2.coords))
                norms.append(norm)
        else:
            norms.append(1)
        return area


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


# # # Precompute domain arguments # # #

pass

# # #

domain_cfg = Domain(
    allowed_area=[
        [0, 0],
        [0, 100],
        [100, 100],
        [100, 0],
        [0, 0],
    ],
    name='main',
    min_poly_num=2,
    max_poly_num=4,
    min_points_num=3,
    max_points_num=15,
    polygon_side=0.0001,
    min_dist_from_boundary=0.0001,
    geometry_is_convex=False,
    geometry_is_closed=True,
    geometry='2D',
)

tuner_cfg = TunerParams(
    tuner_type='optuna',
    n_steps_tune=10,
    hyperopt_dist='uniform',
    verbose=True,
    timeout_minutes=60,
)

opt_params = OptimizationParams(
    domain=domain_cfg,
    tuner_cfg=tuner_cfg,
    n_steps=10,
    pop_size=50,
    postprocess_attempts=3,
    mutation_prob=0.6,
    crossover_prob=0.6,
    mutations=[
        'rotate_poly',
        'resize_poly',
        'add_point',
        'drop_point',
        'add_poly',
        'drop_poly',
        'pos_change_point',
    ],
    selector='tournament_selection',
    mutation_each_prob=[0.125, 0.125, 0.15, 0.35, 0.00, 0.00, 0.25],
    crossovers=[
        'polygon_level',
        'structure_level',
    ],
    crossover_each_prob=[0.0, 1.0],
    postprocess_rules=[
        'not_out_of_bounds',
        'valid_polygon_geom',
        'not_self_intersects',
        'not_too_close_polygons',
        # 'not_overlaps_prohibited',
        'not_too_close_points',
    ],
    extra=5,
    n_jobs=0,
    log_dir='logs',
    run_name='run_name',
    golem_keep_histoy=False,
    golem_genetic_scheme_type='steady_state',
    golem_surrogate_each_n_gen=5,
    objectives=[
        Area(domain_cfg),
        SideCoef(domain_cfg),
    ],
)
