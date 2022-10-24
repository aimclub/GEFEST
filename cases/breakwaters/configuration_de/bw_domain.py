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
fixed_area = [
    [[471, 5], [1335, 2], [1323, 214], [1361, 277], [1395, 327], [1459, 405], [1485, 490], [1449, 521], [1419, 558],
     [1375, 564], [1321, 469], [1248, 318], [1068, 272], [921, 225], [804, 231], [732, 266], [634, 331], [548, 405],
     [485, 482], [424, 569], [381, 625], [310, 662], [271, 684], [244, 706], [203, 708], [182, 647], [214, 638],
     [234, 632], [275, 588], [346, 475], [427, 366], [504, 240], [574, 166], [471, 5]],
    [[652, 1451], [580, 1335], [544, 1253], [468, 1190], [439, 1170], [395, 1150], [378, 1115], [438, 1070],
     [481, 1059], [508, 1076], [539, 1133], [554, 1183], [571, 1244], [594, 1305], [631, 1366], [657, 1414],
     [671, 1449], [652, 1451]]
]
fixed_targets = [[coord_X[26], coord_Y[49]], [coord_X[37], coord_Y[11]], [coord_X[60], coord_Y[5]]]
fixed_poly = [
    [[878, 1433], [829, 1303], [739, 1116], [619, 995], [447, 962], [306, 1004], [254, 1092], [241, 1184],
     [269, 1244],
     [291, 1338], [370, 1450]],
    [[878, 1433], [829, 1303], [739, 1116], [619, 995], [447, 962], [274, 868], [180, 813], [126, 717], [146, 580],
     [203, 480], [249, 469], [347, 471]]
]

# Creation of prohibited structure consist of targets, lines, areas
prohibited_structure = create_prohibited(targets=fixed_targets, fixed_area=fixed_area,
                                         fixed_points=fixed_poly)


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
