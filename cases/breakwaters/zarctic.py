
from shapely.affinity import rotate
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon, LineString, mapping, GeometryCollection
from shapely.ops import split
from shapely.affinity import scale
from shapely.plotting import plot_polygon, plot_line, plot_points
from shapely import get_parts
from polygenerator import random_convex_polygon as rcp
from polygenerator import random_polygon as rp
from polygenerator import random_star_shaped_polygon as rsp
from shapely.geometry import shape, JOIN_STYLE
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.geometry import Polygon as Pol, Point as Poi, Structure
from gefest.core.geometry.datastructs.polygon import PolyID
import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from gefest.core.utils.functions import parse_structs,project_root

pr_root=project_root()

if __name__ == "__main__":
    root = f"{pr_root}/cases/breakwaters/newdata"
    border_geo, landsacpe, res_geo_targets = [root + '\\' + fname for fname in os.listdir(root)]

    with open(border_geo, 'r') as file:
        border_dict = json.load(file)
    border = shape(border_dict['features'][0]['geometry'])
    #plot_polygon(border,color='black')
    #plt.show()
    print(landsacpe)

    with open(res_geo_targets, 'r') as file:
        res_targets = json.load(file)

    # for idx, target in enumerate(res_targets['features']):
    #     # print(idx)
    #     res = shape(target['geometry'])
    #     if isinstance(res, LineString):
    #         plot_line(res)
    #         #pass
    #     if isinstance(res, ShapelyPolygon):
    #         plot_polygon(res)
    #         #pass
    piers = [i for i in res_targets['features'] if (i['properties']['type']=='passenger_pier') or (i['properties']['type'] =='cargo_pier')]
    cargo_piers = [i for i in res_targets['features'] if i['properties']['type'] == 'cargo_pier']
    passenger_pier = [i for i in res_targets['features'] if i['properties']['type'] == 'passenger_pier']
    piers_coords = [x[0] for x in [i['geometry']['coordinates'] for i in piers]]
    piers_max = [max(p, key=lambda i: i[1]) for p in piers_coords]
    water = [i for i in res_targets['features'] if i['properties']['type'] == 'water']
    plt.plot([p[0] for p in piers_max],[p[1] for p in piers_max], color='blue')
    for i in water:
        res = shape(i['geometry'])
        plot_polygon(res,color='yellow')
    # for i in passenger_pier:
    #     res = shape(i['geometry'])
    #     plot_polygon(res,color='green')
    plt.show()

    df = pd.read_csv(landsacpe)
    for idx in range(df.shape[0]):
        color='r'
        if df.iloc[idx][3] in [15,16] and df.iloc[idx][4] in [14,15,16,17,18,19]:
            color = 'y'
        plot_points(ShapelyPoint((df.iloc[idx][1], df.iloc[idx][0])), color=color)
    plt.show()

#    fig = plt.figure()
    data = np.random.rand(10, 10)
    #plt.plot(data)
    #plt.show()
    #sns.heatmap(data, vmax=.8, square=False)

    z = np.loadtxt(f'{pr_root}/cases/breakwaters/swan_model/swan_model/results/HSig_ob_example_20180102.000000_20180103.230000.dat')

    def init():
        sns.heatmap(z[0:32, :], vmin = 0., square=False, cbar=False)

    # def animate(i):
    #     data = z[i*32:(i+1)*32, :]
    #     sns.heatmap(data, vmin=0., square=False, cbar=False)

    #anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1538//32, repeat = False)
    #anim.save('anim.mp4')

    plt.show(block=True)
    def init():
        sns.heatmap(z[0:32, :], vmin = 0., square=False, cbar=False)