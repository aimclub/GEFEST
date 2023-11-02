import copy

import shapely
from shapely.validation import explain_validity
from typing import List

from cases.sound_waves.poly_from_point import poly_from_comsol_txt
from core.algs.postproc.resolve_errors import PolygonNotSelfIntersects
from core.geometry.domain import Domain
from gefest.core.utils.functions import parse_structs,project_root
from gefest.core.geometry import Point, Polygon, Structure
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shapely.plotting import plot_polygon, plot_line, plot_points
import json
import os
import time
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
from gefest.tools.samplers.sampler import Sampler
from gefest.core.geometry.geometry_2d import Geometry2D
from tools.samplers.rand_gen.noise_sampler import NoisedPoly

root = Path(__file__).parent.parent.parent.parent.parent
grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300
coord_X = np.linspace(1, 2, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(1, 2, grid_resolution_y + 1)
border = [
            (min(coord_X), min(coord_Y)),
            (min(coord_X), max(coord_Y)),
            (max(coord_X), max(coord_Y)),
            (max(coord_X), min(coord_Y)),
            (min(coord_X), min(coord_Y)),
        ]
len_x =max(coord_X)- min(coord_X)
len_y =max(coord_Y)- min(coord_Y)
geometry = Geometry2D()

path_=f"{root}/cases/synthetic/syn_gen/Comsol_points/lightning.txt"
#best_poly = poly_from_comsol_txt(path=path_).polygons[0]

best_poly = random_star_shaped_polygon(num_points=4)
best_poly = Polygon([Point(p[0]*min(coord_X)+len_x/2,p[1]*min(coord_Y)+len_y/2) for p in best_poly])
#best_poly.points.append(best_poly.points[0])



plt.ion()
noise = NoisedPoly(init_poly=best_poly,scale=0.01,degrees_to_rotate=10)
for _ in range(200):
    #angle = np.random.randint(-100, 100)
    plt.clf()
    #plt.plot([b[0] for b in border],[b[1] for b in border])
    plt.plot([b[0] for b in [p.coords for p in best_poly.points]],[b[1] for b in [p.coords for p in best_poly.points]])
    #plt.plot([b[0] for b in [p.coords for p in geometry.rotate_poly(best_poly,angle).points]],
    # [b[1] for b in [p.coords for p in geometry.rotate_poly(best_poly,angle).points]])
    noised = noise().points
    plt.plot([b[0] for b in [p.coords for p in noised]],
             [b[1] for b in [p.coords for p in noised]],
             label='generated')
    plt.legend()
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.08)
plt.show()