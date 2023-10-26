import copy
from cases.sound_waves.poly_from_point import poly_from_comsol_txt
from gefest.core.utils.functions import parse_structs,project_root
from gefest.core.geometry import Point, Polygon, Structure
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
from gefest.core.geometry.geometry_2d import Geometry2D
root = Path(__file__).parent.parent.parent.parent.parent
grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300
coord_X = np.linspace(20, 100, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(20, 100, grid_resolution_y + 1)
border = [
            (min(coord_X), min(coord_Y)),
            (min(coord_X), max(coord_Y)),
            (max(coord_X), max(coord_Y)),
            (max(coord_X), min(coord_Y)),
            (min(coord_X), min(coord_Y)),
        ]
geometry = Geometry2D()

path_=f"{root}/cases/synthetic/syn_gen/Comsol_points/lightning.txt"
#best_poly = poly_from_comsol_txt(path=path_).polygons[0]
best_poly = random_star_shaped_polygon(num_points=20)
len_x =max(coord_X)- min(coord_X)
len_y =max(coord_Y)- min(coord_Y)
best_poly = Polygon([Point(p[0]*min(coord_X)+len_x/2,p[1]*min(coord_Y)+len_y/2) for p in best_poly])
def noise_coords(poly,scale=1.0):
    new_poly = Polygon([])
    angle = np.random.randint(-100, 100)

    #poly = [p.coords for p in poly.points]
    sigma_max_x = max(p.coords[0] for p in poly.points)
    sigma_max_y = max(p.coords[1] for p in poly.points)
    sigma_min_x = min(p.coords[0] for p in poly.points)
    sigma_min_y = min(p.coords[1] for p in poly.points)
    max_x = sigma_max_x-sigma_min_x
    max_y = sigma_max_y - sigma_min_y
    sigma = np.random.uniform(min(max_x,max_y)*scale,max(max_x,max_y)*scale)
    x_noise_start = np.random.uniform(-sigma, sigma)
    y_noise_start = np.random.uniform(-sigma, sigma)
    for i,point in enumerate(poly.points):
        if i==0 or i==(len(poly)-1):
            new_poly.points.append(Point(point.coords[0] + x_noise_start, point.coords[1] + y_noise_start))
            continue
        if np.random.uniform(0,1) <0.5:
            new_poly.points.append(point)
        else:
            x_noise = np.random.uniform(-sigma,sigma)
            y_noise = np.random.uniform(-sigma, sigma)
            new_poly.points.append(Point(point.coords[0] + x_noise, point.coords[1] + y_noise))
    if np.random.uniform(0, 1) < 0.5:
        x_scale= np.random.uniform(0.5, 2)
        y_scale = np.random.uniform(0.5, 2)
        new_poly=geometry.resize_poly(new_poly,x_scale,y_scale)
    if np.random.uniform(0, 1) < 0.5:
        new_poly = geometry.rotate_poly(new_poly,angle)
    return new_poly

plt.ion()
for _ in range(200):
    angle = np.random.randint(-100, 100)
    plt.clf()
    plt.plot([b[0] for b in border],[b[1] for b in border])
    plt.plot([b[0] for b in [p.coords for p in best_poly.points]],[b[1] for b in [p.coords for p in best_poly.points]])
    #plt.plot([b[0] for b in [p.coords for p in geometry.rotate_poly(best_poly,angle).points]],
    # [b[1] for b in [p.coords for p in geometry.rotate_poly(best_poly,angle).points]])
    noised = noise_coords(best_poly, scale=0.05).points
    plt.plot([b[0] for b in [p.coords for p in noised]],
             [b[1] for b in [p.coords for p in noised]])
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.06)
plt.show()