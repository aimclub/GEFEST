import numpy as np
from gefest.core.geometry import Point, Polygon, Structure
import matplotlib.pyplot as plt
import pickle
from functools import partial
import json
from uuid import uuid4
import numpy as np
from shapely.geometry import shape
from hyperopt import hp
from gefest.core.opt.operators.selections import tournament_selection,roulette_selection
from gefest.core.algs.postproc.resolve_errors import Rules, postprocess
from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.geometry.utils import get_random_structure
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.core.opt.operators.crossovers import (
    panmixis,
    polygon_level_crossover,
    structure_level_crossover,
)
from cases.sound_waves.microphone_points import Microphone
from gefest.core.opt.operators.mutations import (
    add_point,
    add_poly,
    drop_point,
    drop_poly,
    pos_change_point_mutation,
    resize_poly,
    rotate_poly,
)
from gefest.core.opt.tuning.tuner import GolemTuner
from gefest.core.structure.prohibited import create_prohibited
from gefest.core.viz.struct_vizualizer import GIFMaker
from gefest.tools.estimators.simulators.swan.swan_interface import Swan
from gefest.tools.fitness import Fitness
from gefest.tools.optimizers.GA.base_GA import BaseGA
from pathlib import Path
#from poly_from_point import poly_from_comsol_txt
#Parsing data for coordinates
root_path = Path(__file__).parent.parent.parent
with open(f'{root_path}/cases/breakwaters/newdata/result_PwiOsA2HE2igZUel.geojson', 'r') as file:
    res_list = json.load(file)
with open(f'{root_path}/cases/breakwaters/newdata/border_PwiOsA2HE2igZUel.geojson', 'r') as file:
    border_dict = json.load(file)
border = shape(border_dict['features'][0]['geometry'])
water = [i for i in res_list['features'] if i['properties']['type'] =='water']
water_coord =[p['geometry']['coordinates'] for p in water]
cargo_piers = [i for i in res_list['features'] if i['properties']['type'] =='cargo_pier']
passenger_pier = [i for i in res_list['features'] if i['properties']['type'] =='passenger_pier']
piers = [i for i in res_list['features'] if (i['properties']['type']=='passenger_pier') or (i['properties']['type'] =='cargo_pier')]
piers_coords = [x[0] for x in [i['geometry']['coordinates'] for i in piers]]
piers_line = [max(p,key=lambda i: i[1]) for p in piers_coords]
unique_types = np.unique([i['properties']['type'] for i in res_list['features']])
allow_water = [i for i in water_coord[0][0] if (i[0]>74.8) and (i[1]<67.942) and (i[1]>67.915)]
###########


# pre domain params
allow_area = [[74.80,67.92],[74.80,67.94]]+allow_water +[[74.80,67.92]]
grid_resolution_x = 17  # Number of points on x-axis
grid_resolution_y = 31  # Number of points on y-axis
coord_Y = np.linspace(min([p[1] for p in allow_area]), max([p[1] for p in allow_area]), grid_resolution_y + 1)  # X coordinate for spatial grid
coord_X = np.linspace(min([p[0] for p in allow_area]), max([p[0] for p in allow_area]), grid_resolution_x + 1)
grid = [grid_resolution_x, grid_resolution_y]  # points grid
fixed_area = None#[[[45, 55], [55, 55], [55, 45], [45, 45], [45, 55]]]


def load_file_from_path(path: str):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        f.close()
    return file


if __name__ == '__main__':


    #  in the future all model can be loaded from configs

    #  domain configuration
    geometry = Geometry2D(is_closed=False, is_convex=True)
    prohibited = create_prohibited(1, [], [], fixed_area=fixed_area)
    domain = Domain(
        allowed_area=[
            (min(coord_X), min(coord_Y)),
            (min(coord_X), max(coord_Y)),
            (max(coord_X), max(coord_Y)),
            (max(coord_X), min(coord_Y)),
            (min(coord_X), min(coord_Y)),
        ],
        geometry=geometry,
        max_poly_num=1,
        min_poly_num=1,
        max_points_num=10,
        min_points_num=3,
        prohibited_area=prohibited,
    )




path_to_input = 'INPUT'
polygons = [Polygon(points=[Point(x=74.81599090607997, y=67.934161908353), Point(x=74.81362548359442, y=67.938407779675), Point(x=74.80617799768942, y=67.93715592872393), Point(x=74.80258026739142, y=67.93554092015472), Point(x=74.80152657396846, y=67.92493736227621), Point(x=74.8027876512988, y=67.92394344756349), Point(x=74.80902484369516, y=67.9257106962192), Point(x=74.81006215245631, y=67.9261835264202)])]
with open(path_to_input, 'r') as file_to_read:
    content_read = file_to_read.read()

    for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '
    num_of_polygons = len(polygons)
    for j, poly in enumerate(polygons):
        num_of_points = len(2 * poly.points)
        points = np.array([p.coords[:2] for p in poly.points])
        individ = points.reshape(-1)
        for i, gen in enumerate(individ):
            for_input += '{:.6f}'.format(gen) + ', '

        if j == (num_of_polygons - 1):
            for_input += '\n$optline'
        else:
            for_input += '\nOBSTACLE TRANSM 0. REFL 0. LINE '

    content_to_replace = for_input
    content_write = content_read.replace(
        content_read[content_read.find('OBSTACLE') : content_read.rfind('$optline')+8],
        content_to_replace,
    )

with open(path_to_input, 'w') as file_to_write:
    file_to_write.write(content_write)