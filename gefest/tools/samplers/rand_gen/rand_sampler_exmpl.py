# Example to generate similar synthetic geometry.
# In the end show animation of revers mutation action.
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from core.geometry import Point, Polygon
from polygenerator import random_star_shaped_polygon
from tools.samplers.rand_gen.noise_sampler import NoisedPoly

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.postproc.rules import PolygonNotSelfIntersects
from gefest.tools.utils import poly_from_comsol_txt

root = Path(__file__).parent.parent.parent.parent.parent
grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300
coord_x = np.linspace(20, 100, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_y = np.linspace(20, 100, grid_resolution_y + 1)
border = [
    (min(coord_x), min(coord_y)),
    (min(coord_x), max(coord_y)),
    (max(coord_x), max(coord_y)),
    (max(coord_x), min(coord_y)),
    (min(coord_x), min(coord_y)),
]
len_x = max(coord_x) - min(coord_x)
len_y = max(coord_y) - min(coord_y)
geometry = Geometry2D()

POLYGON_FROM_TXT = False  # Choose mode to obtain polygon

if POLYGON_FROM_TXT:
    path_ = f'{root}/cases/synthetic/syn_gen/Comsol_points/lightning.txt'  # Load poly from txt file
    best_poly = poly_from_comsol_txt(path=path_).polygons[0]
else:
    best_poly = random_star_shaped_polygon(num_points=77)  # Generate random polygon
    best_poly = Polygon(
        [Point(p[0] * min(coord_x) + len_x / 2, p[1] * min(coord_y) + len_y / 2)
         for p in best_poly]
    )
    best_poly.points.append(best_poly.points[0])

plt.ion()
noise = NoisedPoly(
    init_poly=best_poly,
    scale=0.03,
    degrees_to_rotate=30,
    rules=[PolygonNotSelfIntersects()],
    resize_scale=[0.5, 1.75],
)

for _ in range(200):
    # angle = np.random.randint(-100, 100)
    plt.clf()
    plt.plot([b[0] for b in border], [b[1] for b in border])
    plt.plot(
        [b[0] for b in [p.coords for p in best_poly.points]],
        [b[1] for b in [p.coords for p in best_poly.points]],
    )
    # plt.plot([b[0] for b in [p.coords for p in geometry.rotate_poly(best_poly,angle).points]],
    # [b[1] for b in [p.coords for p in geometry.rotate_poly(best_poly,angle).points]])
    noised = noise().points
    plt.plot(
        [b[0] for b in [p.coords for p in noised]],
        [b[1] for b in [p.coords for p in noised]],
        label='generated',
    )
    plt.legend()
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.08)

plt.show()
