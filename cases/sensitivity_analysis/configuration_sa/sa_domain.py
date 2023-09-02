import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.prohibited import create_prohibited

# ------------
# USER-DEFINED CONFIGURATION OF DOMAIN FOR BREAKWATERS TASK
# ------------

grid_resolution_x = 83  # Number of points on x-axis
grid_resolution_y = 58  # Number of points on y-axis
coord_X = np.linspace(0, 2075, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(0, 1450, grid_resolution_y + 1)  # Y coordinate for spatial grid

grid = [grid_resolution_x, grid_resolution_y]  # points grid
targets = [[49, 26], [11, 37], [5, 60]]  # grid coordinates of considered targets

"""
Prohibited objects
"""
fixed_targets = [[coord_X[26], coord_Y[49]], [coord_X[37], coord_Y[11]], [coord_X[60], coord_Y[5]]]

# Creation of prohibited structure consist of targets, lines, areas
prohibited_structure = create_prohibited(targets=fixed_targets)


def configurate_domain(poly_num: int,
                       points_num: int,
                       is_closed: bool):
    # ------------
    # GEFEST domain based on user-defined configuration_de
    # ------------
    if is_closed:
        min_points_num = 3
    else:
        min_points_num = 2

    geometry = Geometry2D(is_closed=is_closed)
    domain = Domain(allowed_area=[(min(coord_X), min(coord_Y)),
                                  (min(coord_X), max(coord_Y)),
                                  (max(coord_X), max(coord_Y)),
                                  (max(coord_X), min(coord_Y)),
                                  (min(coord_X), min(coord_Y))],
                    prohibited_area=prohibited_structure,
                    geometry=geometry,
                    max_poly_num=poly_num,
                    min_poly_num=1,
                    max_points_num=points_num,
                    min_points_num=min_points_num,
                    is_closed=is_closed)
    task_setup = Setup(domain=domain)

    return domain, task_setup
