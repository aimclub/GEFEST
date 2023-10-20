from gefest.core.utils.functions import parse_structs,project_root
from gefest.core.geometry import Point, Polygon, Structure
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
def find_grid_point(x,y,max_x,min_x,max_y,min_y,grid):
    step_for_point_x = (max_x - min_x)/grid[0]
    step_for_point_y = (max_y - min_y)/grid[1]
    coord = [min_x+step_for_point_x*x,min_y+step_for_point_y*y]
    return coord
grid_resolution_x = 17  # Number of points on x-axis
grid_resolution_y = 31
grid = [grid_resolution_x,grid_resolution_y]

border = [[74.7,74.83,74.83,74.7],[67.88,67.88,67.96,67.96]]
all = [Point(x=74.8-0.04, y=67.9174616), Point(x=74.8-0.04, y=67.9410473), Point(x=74.8192817-0.04, y=67.9410473), Point(x=74.8192817-0.04, y=67.9174616), Point(x=74.8-0.04, y=67.9174616)]
root = Path(__file__).parent.parent.parent

poly = parse_structs(f'{root}/cases/breakwaters/logs/run_name_2023-10-20_11_40_08/00003.log')[0].polygons
poly_init = parse_structs(f'{root}/cases/breakwaters/logs/run_name_2023-10-19_17_11_17/00000_init.log')[-1].polygons
z = np.loadtxt(f'{root}/cases/breakwaters/ob2/results/HSig_ob_example.dat')

for p in poly:
    poly_cd = [i.coords for i in p.points]
    plt.plot([x[0]/500 for x in poly_cd], [y[1]/500 for y in poly_cd])

all_cd = [i.coords for i in all]
target = [11,32-16]
coord_target = find_grid_point(target[0],target[1],74.83,74.7,67.96,67.88,grid)
plt.scatter(coord_target[0],coord_target[1])
plt.plot([x[0] for x in all_cd], [y[1] for y in all_cd])
plt.plot(border[0],border[1])
plt.show()
x = np.arange(0, grid_resolution_x+1, 1)  # len = 11
y = np.arange(0, grid_resolution_y+1, 1)

plt.pcolormesh(x,y,z[:32])
plt.show()