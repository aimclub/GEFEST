from pathlib import Path

import numpy as np

from cases.sound_waves.microphone_points import Microphone
from cases.sound_waves.poly_from_point import poly_from_comsol_txt
from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import get_random_structure
from gefest.core.opt.objective.objective import Objective
from gefest.tools.estimators.simulators.sound_wave.sound_interface import (
    SoundSimulator,
    generate_map,
)

# # # Metrics # # #


class SoundFieldFitness(Objective):
    """Evaluates sound pressure level difference with reference."""

    def __init__(self, domain, estimator, path_best_struct=None):
        super().__init__(domain, estimator)
        self.path_best_struct = path_best_struct

        if self.path_best_struct is None:
            print('Please, set up the best spl matrix into configuration.')
            print('The best structure will be generated randomly.')
            rnd_structure = get_random_structure(domain)
            best_spl = generate_map(domain, rnd_structure)
        else:
            best_structure = poly_from_comsol_txt(path_best_struct)
            best_spl = self.estimator(best_structure)
            best_spl = np.nan_to_num(best_spl, nan=0, neginf=0, posinf=0)
            micro = Microphone(matrix=best_spl).array()
            best_spl = np.concatenate(micro[1])

        self.best_spl = best_spl

    def _evaluate(self, ind: Structure):
        spl = self.estimator(ind)
        current_spl = np.nan_to_num(spl, nan=0, neginf=0, posinf=0)
        micro = Microphone(matrix=current_spl).array()
        current_spl = np.concatenate(micro[1])
        l_f = np.sum(np.abs(self.best_spl - current_spl))
        return l_f


# # # Precompute domain arguments # # #

pass

# # #

domain_cfg = Domain(
    allowed_area=[
        [0, 0],
        [0, 120],
        [120, 120],
        [120, 0],
        [0, 0],
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
        SoundFieldFitness(
            domain_cfg,
            SoundSimulator(domain_cfg, 10, None),
            str(Path(__file__).parent) + '\\figures\\bottom_square.txt',
        ),
    ],
)
