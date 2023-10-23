import copy

from gefest.core.utils.functions import parse_structs,project_root
from gefest.core.geometry import Point, Polygon, Structure
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shapely.plotting import plot_polygon, plot_line, plot_points
import json
import os
import pandas as pd
from shapely.geometry import shape
from shapely.geometry import Polygon as ShapelyPolygon, LineString
def find_grid_point(x,y,max_x,min_x,max_y,min_y,grid):
    step_for_point_x = (max_x - min_x)/grid[0]
    step_for_point_y = (max_y - min_y)/grid[1]
    coord = [min_x+step_for_point_x*x,min_y+step_for_point_y*y]
    return coord

def scale_points(struct:Structure,scale=1):
    new_struct = copy.deepcopy(struct)
    point_array =[]
    for i,poly in enumerate(struct.polygons):
        point_array.append([[point.coords[0]/scale,point.coords[1]/scale] for point in poly])
        new_struct.polygons[i].points = [Point(point.coords[0]/scale,point.coords[1]/scale) for point in poly]
    return new_struct,point_array

grid_resolution_x = 17  # Number of points on x-axis
grid_resolution_y = 31
grid = [grid_resolution_x,grid_resolution_y]

border = [[74.7,74.83,74.83,74.7],[67.88,67.88,67.96,67.96]]
all = [Point(x=74.8-0.04, y=67.9174616), Point(x=74.8-0.04, y=67.9410473), Point(x=74.8192817-0.04, y=67.9410473), Point(x=74.8192817-0.04, y=67.9174616), Point(x=74.8-0.04, y=67.9174616)]
root = Path(__file__).parent.parent.parent

pars = [parse_structs(f'{root}/cases/breakwaters/logs/run_name_2023-10-21_13_56_24/00038.log')[14],
        parse_structs(f'{root}/cases/breakwaters/logs/run_name_2023-10-21_13_56_24/00037.log')[0],
        parse_structs(f'{root}/cases/breakwaters/logs/run_name_2023-10-20_16_08_24/00010.log')[4]
        ]

poly = pars[0].polygons
scaled_point_array={}
scaled_dict = {}
for i,parses in enumerate(pars):
    scaled_dict[i],arr = scale_points(parses,scale=500)
    scaled_point_array[f'Polygon_{i}'] = arr
# with open(f'{root}/cases/breakwaters/coords/polys.json', 'w', encoding='utf-8') as f:
#     json.dump(scaled_point_array, f, ensure_ascii=False, indent=4)

poly_init = parse_structs(f'{root}/cases/breakwaters/logs/run_name_2023-10-19_17_11_17/00000_init.log')[-1].polygons
z = np.loadtxt(f'{root}/cases/breakwaters/ob2/results/HSig_ob_example.dat')
shift = 0.03

all_cd = [i.coords for i in all]
target = [11,32-16]
targets = [[11,32-i] for i in [10,12,14,16]]
for t in targets:
    coord_target = find_grid_point(t[0],t[1],74.83,74.7,67.96,67.88,grid)
coord_target = [find_grid_point(t[0],t[1],74.83,74.7,67.96,67.88,grid) for t in targets]
plt.scatter([x[0]+shift for x in coord_target],[y[1] for y in coord_target],c='Red', marker='*',)

#plt.plot([x[0]+shift for x in all_cd], [y[1] for y in all_cd])#Allow area
#plt.plot(border[0],border[1])#Border

#plt.show()
x = np.arange(0, grid_resolution_x+1, 1)  # len = 11
y = np.arange(0, grid_resolution_y+1, 1)




root = f"{root}/cases/breakwaters/newdata"
border_geo, landsacpe, res_geo_targets = [root + '\\' + fname for fname in os.listdir(root)]

with open(res_geo_targets, 'r') as file:
    res_targets = json.load(file)


for p in poly:
    poly_cd = [i.coords for i in p.points]#[p.points[0:1]+p.points[4::]][0]
    plt.plot([x[0]/500+shift for x in poly_cd], [y[1]/500 for y in poly_cd])
    with open(res_geo_targets, 'r') as file:
        res_targets = json.load(file)

    for idx, target in enumerate(res_targets['features']):
        # print(idx)
        res = shape(target['geometry'])
        if isinstance(res, LineString):
            plot_line(res)
            #pass
        if isinstance(res, ShapelyPolygon):
            plot_polygon(res)
plt.title(f'Wave Height:{pars.fitness[0]} \n BW length: {pars.fitness[1]}')
plt.legend()
piers = [i for i in res_targets['features'] if (i['properties']['type']=='passenger_pier') or (i['properties']['type'] =='cargo_pier')]
cargo_piers = [i for i in res_targets['features'] if i['properties']['type'] == 'cargo_pier']
passenger_pier = [i for i in res_targets['features'] if i['properties']['type'] == 'passenger_pier']
piers_coords = [x[0] for x in [i['geometry']['coordinates'] for i in piers]]
piers_max = [max(p, key=lambda i: i[1]) for p in piers_coords]
water = [i for i in res_targets['features'] if i['properties']['type'] == 'water']
plt.plot([p[0] for p in piers_max],[p[1] for p in piers_max], color='blue')
for i in water:
    res = shape(i['geometry'])
    plot_polygon(res,color='black')
# for i in passenger_pier:
#     res = shape(i['geometry'])
#     plot_polygon(res,color='green')
plt.show()
plt.pcolormesh(x,y,z[:32])
plt.show()