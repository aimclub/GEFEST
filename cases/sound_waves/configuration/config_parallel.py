from copy import deepcopy
from pathlib import Path

import numpy as np

from cases.sound_waves.microphone_points import Microphone
from gefest.tools.utils import poly_from_comsol_txt
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
DAMPING = 1 - 0.001
CA2 = 0.5
INITIAL_P = 200
MAX_PRESSURE = INITIAL_P / 2
MIN_PRESSURE = -INITIAL_P / 2

import itertools
from numpy.core.umath import pi
class SoundFieldFitness(Objective):
    """Evaluates sound pressure level difference with reference."""

    def __init__(self, domain, estimator, best_spl, duration):
        super().__init__(domain, estimator)
        self.best_spl = best_spl
        self.omega = 3 / (2 * pi)
        self.iteration = 0
        self.map_size = (round(1.2 * domain.max_y), round(1.2 * domain.max_x))
        self.size_y, self.size_x = self.map_size
        self.duration = duration
        # obstacle_map handling

        # Source position is the center of the map
        self.s_y = self.size_y // 2
        self.s_x = self.size_x // 2

    def update_velocity(self, velocities, pressure, obstacle_map):
        """Update the velocity field based on Komatsuzaki's transition rules."""
        V = velocities
        P = pressure
        for i, j in itertools.product(range(self.size_y), range(self.size_x)):
            if obstacle_map[i, j] == 1:
                V[i, j, 0:4] = 0.0
                continue

            V[i, j, 0] = V[i, j, 0] + P[i, j] - P[i - 1, j] if i > 0 else P[i, j]
            V[i, j, 1] = V[i, j, 1] + P[i, j] - P[i, j + 1] if j < self.size_x - 1 else P[i, j]
            V[i, j, 2] = V[i, j, 2] + P[i, j] - P[i + 1, j] if i < self.size_y - 1 else P[i, j]
            V[i, j, 3] = V[i, j, 3] + P[i, j] - P[i, j - 1] if j > 0 else P[i, j]

        return V

    def step(self, velocities, pressure, obstacle_map):
        """Perform a simulation step, upadting the wind an pressure fields."""
        pressure[self.s_y, self.s_x] = INITIAL_P * np.sin(self.omega * self.iteration)
        velocities = self.update_velocity(velocities, pressure, obstacle_map)
        pressure = self.update_perssure(pressure, velocities)
        self.iteration += 1
        return velocities, pressure

    def update_perssure(self, pressure, velocities):
        """Update the pressure field based on Komatsuzaki's transition rules."""
        pressure -= CA2 * DAMPING * np.sum(velocities, axis=2)
        return pressure

    def spl(self, pressure_hist, integration_interval=60):
        """Computes the sound pressure level map.

        https://en.wikipedia.org/wiki/Sound_pressure#Sound_pressure_level

        Args:
            integration_interval (int): interval over which the rms pressure
                                        is computed, starting from the last
                                        simulation iteration backwards.

        Returns:
            spl (np.array): map of sound pressure level (dB).
        """
        p0 = 20 * 10e-6  # Pa
        if integration_interval > pressure_hist.shape[0]:
            integration_interval = pressure_hist.shape[0]

        rms_p = np.sqrt(np.mean(np.square(pressure_hist[-integration_interval:-1]), axis=0))

        rms_p[rms_p == 0.0] = 0.000000001
        matrix_db = 20 * np.log10(rms_p / p0)
        return matrix_db

    def _evaluate(self, ind: Structure):
        self.iteration = 0
        obstacle_map = np.zeros((self.size_y, self.size_x))
        pressure = np.zeros((self.size_y, self.size_x))
        pressure_hist = np.zeros((self.duration, self.size_y, self.size_x))
        velocities = np.zeros((self.size_y, self.size_x, 4))

        for iteration in range(self.duration):
            pressure_hist[iteration] = deepcopy(pressure)
            velocities, pressure = self.step(velocities, pressure, obstacle_map)
        # best_spl = self._reference_spl(sim)
        spl = self.spl(pressure_hist)

        current_spl = np.nan_to_num(spl, nan=0, neginf=0, posinf=0)
        micro = Microphone(matrix=current_spl).array()
        current_spl = np.concatenate(micro[1])
        l_f = np.sum(np.abs(deepcopy(self.best_spl) - current_spl))
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
    n_steps_tune=25,
    hyperopt_dist='uniform',
    verbose=True,
    timeout_minutes=60,
)


best_structure = poly_from_comsol_txt(str(Path(__file__).parent) + '\\figures\\bottom_square.txt')
best_spl = SoundSimulator(domain_cfg, 50, None)(best_structure)
best_spl = np.nan_to_num(best_spl, nan=0, neginf=0, posinf=0)
micro = Microphone(matrix=best_spl).array()
best_spl = np.concatenate(micro[1])

opt_params = OptimizationParams(
    optimizer='gefest_ga',
    domain=domain_cfg,
    tuner_cfg=tuner_cfg,
    n_steps=10,
    pop_size=10,
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
    estimation_n_jobs=1,
    n_jobs=-1,
    log_dir='logs',
    run_name='run_name',
    golem_keep_histoy=False,
    golem_genetic_scheme_type='steady_state',
    golem_surrogate_each_n_gen=5,
    objectives=[
        SoundFieldFitness(
            domain_cfg,
            None,
            best_spl,
            50
        ),
    ],
)
