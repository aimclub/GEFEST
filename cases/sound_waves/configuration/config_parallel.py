from functools import partial
from pathlib import Path

import numpy as np

from cases.sound_waves.microphone_points import Microphone
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
from gefest.tools.tuners.utils import percent_edge_variance
from gefest.tools.utils import poly_from_comsol_txt


class SoundFieldFitness(Objective):
    """Evaluates sound pressure level difference with reference."""

    def __init__(self, domain, estimator, path_best_struct=None, micro_slice=-1):
        super().__init__(domain, estimator)
        self.path_best_struct = path_best_struct
        self.micro_slice = micro_slice
        if path_best_struct is None:
            print('please, set up the best spl matrix into configuration')
            print('the best structure will be generated randomly')
            rnd_structure = get_random_structure(domain)
            best_spl = generate_map(domain, rnd_structure)
        else:
            best_structure = poly_from_comsol_txt(path_best_struct)
            best_spl = self.estimator(best_structure)
            best_spl = np.nan_to_num(best_spl, nan=0, neginf=0, posinf=0)
            micro = Microphone(matrix=best_spl).array()
            best_spl = np.concatenate(micro[micro_slice])

        self.best_spl = best_spl

    def _evaluate(self, ind: Structure):

        spl = self.estimator(ind)
        current_spl = np.nan_to_num(spl, nan=0, neginf=0, posinf=0)
        micro = Microphone(matrix=current_spl).array()
        current_spl = np.concatenate(micro[self.micro_slice])
        l_f = np.sum(np.abs(self.best_spl - current_spl))
        return l_f / len(current_spl)


# # # domain pre-computation

pass

# # #

domain_cfg = Domain(
    allowed_area=[
        [20, 20],
        [20, 100],
        [100, 100],
        [100, 20],
        [20, 20],
    ],
    name='main',
    min_poly_num=1,
    max_poly_num=4,
    min_points_num=3,
    max_points_num=16,
    polygon_side=0.001,
    min_dist_from_boundary=0.001,
    geometry_is_convex=True,
    geometry_is_closed=True,
    geometry='2D',
)


tuner_cfg = TunerParams(
    tuner_type='sequential',
    n_steps_tune=50,
    hyperopt_dist='uniform',
    verbose=True,
    variacne_generator=partial(percent_edge_variance, percent=0.5),
    timeout_minutes=30,
)


opt_params = OptimizationParams(
    optimizer='gefest_ga',
    domain=domain_cfg,
    tuner_cfg=tuner_cfg,
    n_steps=100,
    pop_size=100,
    postprocess_attempts=3,
    mutation_prob=0.9,
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
    mutation_each_prob=[0.125, 0.125, 0.25, 0.25, 0.0, 0.0, 0.25],
    crossovers=[
        'polygon_level',
        'structure_level',
    ],
    crossover_each_prob=[1.0, 0.0],
    postprocess_rules=[
        'not_out_of_bounds',
        'valid_polygon_geom',
        'not_self_intersects',
        'not_too_close_polygons',
        # 'not_overlaps_prohibited',
        'not_too_close_points',
    ],
    extra=5,
    estimation_n_jobs=-1,
    n_jobs=-1,
    log_dir='logs/tuners_exp',
    run_name='roulette_1_obj',
    golem_keep_histoy=True,
    golem_genetic_scheme_type='steady_state',
    golem_surrogate_each_n_gen=5,
    objectives=[
        SoundFieldFitness(
            domain_cfg,
            SoundSimulator(domain_cfg, 200),
            str(Path(__file__).parent) + '\\figures\\bottom_square.txt',
            -1,
        ),
    ],
)
