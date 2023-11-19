import itertools
from copy import deepcopy

import numpy as np
from numpy import ndarray
from numpy.core.umath import pi
from skimage.draw import polygon as skipolygon
from skimage.draw import random_shapes

from gefest.core.geometry import Structure
from gefest.tools import Estimator

# initial values
DAMPING = 1 - 0.001
CA2 = 0.5
INITIAL_P = 200
MAX_PRESSURE = INITIAL_P / 2
MIN_PRESSURE = -INITIAL_P / 2


def generate_map(domain, structure):
    """Generates obstacke map according to polygons in structure inside of domain borders.

    Args:
        domain(Domain): shape of the map to generate
        structure (Structure): structure for shaping obstacles

    Returns:
        obstacle_map (np.array): array shaped as domain area, containing polygons as obstacles.

    """
    map_size = (round(1.2 * domain.max_y), round(1.2 * domain.max_x))
    observed_structure = deepcopy(structure)
    obstacle_map = np.zeros(map_size)

    for polygon in observed_structure.polygons:
        r_coords = [point.y for point in polygon.points]
        c_coords = [point.x for point in polygon.points]

        rr, cc = skipolygon(r_coords, c_coords)

        obstacle_map[rr, cc] = 1

    return obstacle_map


def generate_random_map(map_size: tuple[int, int], random_seed: int):
    """Randomly generate an array of zeros (free media) and ones (obstacles).

    The obstacles have basic geometric shapes.

    Args:
        map_size(tuple): shape of the map to generate
        random_seed (int): random seed for random generation of obstacles

    Returns:
        random_map (np.array): array shaped as map_size, containing random obstacles
    """
    result = random_shapes(
        map_size,
        intensity_range=(0, 60),
        min_size=8,
        max_size=15,
        min_shapes=2,
        max_shapes=10,
        num_channels=1,
        random_seed=random_seed,
        allow_overlap=False,
    )
    # result is a tuple consisting of
    # # (1) the image with the generated shapes
    # # (2) a list of label tuples with the kind of shape
    # (e.g. circle, rectangle) and ((r0, r1), (c0, c1)) coordinates.
    obstacle_map, _ = result
    # Force free media in a square of 20x20 at the center of the map
    width_center = map_size[0] // 2
    length_center = map_size[1] // 2
    obstacle_map[
        width_center - 20 : width_center + 21,
        length_center - 20 : length_center + 21,
    ] = 255
    free_media = obstacle_map == 255
    # Obstacles = 1, free media = 0
    obstacles = obstacle_map == 0
    obstacle_map[free_media] = 0
    obstacle_map[obstacles] = 1
    return obstacle_map


class SoundSimulator(Estimator):
    """Class for the configuration and simulation of sound propagation in a map with obstacles.

    Adapted from https://github.com/Alexander3/wave-propagation
    Based on Komatsuzaki T. "Modelling of Incident Sound Wave Propagation
    around Sound Barriers Using Cellular Automata" (2012)

    Attributes:
        map_size (tuple): size of the map
        obstacle_map (np.array): free media = 0, obstacles = 1. If the given
            shape is different from map_size, ignore the parameters and
            generate a map with no obstacles.
        duration (int): duration (in seconds) of the simulation.
        size_x (int): number of cols in the grid.
        size_y (int): number of cols in the grid.
        pressure (np.array): pressure field at current iteration.
        pressure_hist (np.array): history of all simulated pressure fields.
        _velocities (np.array): velocity field at current iteration.
    """

    def __init__(self, domain, duration=200, obstacle_map=None):
        self.omega = 3 / (2 * pi)
        self.iteration = 0
        self.domain = domain
        self.map_size = (round(1.2 * domain.max_y), round(1.2 * domain.max_x))
        self.size_y, self.size_x = self.map_size
        self.duration = duration
        # obstacle_map handling
        if (
            obstacle_map is not None
            and (obstacle_map.shape[0], obstacle_map.shape[1]) == self.map_size
        ):
            print('** Map Accepted **')
            self.obstacle_map = obstacle_map
        elif obstacle_map is not None and obstacle_map.shape != self.map_size:
            print('** Map size denied **')
            self.obstacle_map = np.zeros((self.size_y, self.size_x))
        else:
            self.obstacle_map = np.zeros((self.size_y, self.size_x))
        # Source position is the center of the map
        self.s_y = self.size_y // 2
        self.s_x = self.size_x // 2
        self.pressure = np.zeros((self.size_y, self.size_x))
        self.pressure_hist = np.zeros((self.duration, self.size_y, self.size_x))
        # outflow velocities from each cell
        self._velocities = np.zeros((self.size_y, self.size_x, 4))

    def update_velocity(self):
        """Update the velocity field based on Komatsuzaki's transition rules."""
        V = self._velocities
        P = self.pressure
        for i, j in itertools.product(range(self.size_y), range(self.size_x)):
            if self.obstacle_map[i, j] == 1:
                V[i, j, 0:4] = 0.0
                continue

            V[i, j, 0] = V[i, j, 0] + P[i, j] - P[i - 1, j] if i > 0 else P[i, j]
            V[i, j, 1] = V[i, j, 1] + P[i, j] - P[i, j + 1] if j < self.size_x - 1 else P[i, j]
            V[i, j, 2] = V[i, j, 2] + P[i, j] - P[i + 1, j] if i < self.size_y - 1 else P[i, j]
            V[i, j, 3] = V[i, j, 3] + P[i, j] - P[i, j - 1] if j > 0 else P[i, j]

    def update_perssure(self):
        """Update the pressure field based on Komatsuzaki's transition rules."""
        self.pressure -= CA2 * DAMPING * np.sum(self._velocities, axis=2)

    def step(self):
        """Perform a simulation step, upadting the wind an pressure fields."""
        self.pressure[self.s_y, self.s_x] = INITIAL_P * np.sin(self.omega * self.iteration)
        self.update_velocity()
        self.update_perssure()
        self.iteration += 1

    def spl(self, integration_interval=60):
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
        if integration_interval > self.pressure_hist.shape[0]:
            integration_interval = self.pressure_hist.shape[0]

        rms_p = np.sqrt(np.mean(np.square(self.pressure_hist[-integration_interval:-1]), axis=0))

        rms_p[rms_p == 0.0] = 0.000000001
        matrix_db = 20 * np.log10(rms_p / p0)
        return matrix_db

    def run(self):
        """Runs soun estimation."""
        for iteration in range(self.duration):
            self.pressure_hist[iteration] = deepcopy(self.pressure)
            self.step()

    def estimate(self, structure: Structure) -> ndarray:
        """Estimates sound pressule level for provided structure.

        Args:
            structure (Structure): optimized structure

        Returns:
            ndarray: map of sound pressure level (dB)
        """
        self.obstacle_map = generate_map(self.domain, structure)
        self.run()
        spl = self.spl()

        self.pressure = np.zeros((self.size_y, self.size_x))
        self.pressure_hist = np.zeros((self.duration, self.size_y, self.size_x))
        self._velocities = np.zeros((self.size_y, self.size_x, 4))

        return spl
