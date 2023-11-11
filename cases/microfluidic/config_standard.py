from pathlib import Path

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective
from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol


# # # Metrics # # #
class ComsolMetric(Objective):
    """Comsol metric."""
    def __init__(self, domain, estimator):
        super().__init__(domain, estimator)

    def _evaluate(self, ind: Structure):
        return self.estimator(ind)


# # # Precompute domain arguments # # #
pass
# # #

domain_cfg = Domain(
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
    name='main',
    min_poly_num=1,
    max_poly_num=1,
    min_points_num=3,
    max_points_num=15,
    polygon_side=0.0001,
    min_dist_from_boundary=0.0001,
    geometry_is_convex=True,
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
    optimizer='gefest_ga',
    domain=domain_cfg,
    tuner_cfg=tuner_cfg,
    n_steps=50,
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
    n_jobs=-1,
    log_dir='logs',
    run_name='run_name',
    golem_keep_histoy=False,
    golem_genetic_scheme_type='steady_state',
    golem_surrogate_each_n_gen=5,
    objectives=[
        ComsolMetric(
            domain_cfg,
            Comsol(str(Path(__file__).parent) + '\\microfluid_file\\rbc-trap-setup.mph'),
        ),
    ],
)
