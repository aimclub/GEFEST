import json
from gefest.core.utils.functions import parse_structs,project_root
from shapely.geometry import shape, JOIN_STYLE
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.geometry import Polygon as Pol, Point as Poi, Structure
from gefest.core.geometry.datastructs.polygon import PolyID
from shapely.plotting import plot_polygon, plot_line, plot_points
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

    with open(res_geo_targets, 'r') as file:
        res_geo_targets_dict = json.load(file)
    border = shape(border_dict['features'][0]['geometry'])
    plot_polygon(border)

plt.show()
pr_root=project_root()
#data = json.loads(f"{pr_root}/cases/breakwaters/newdata/border_PwiOsA2HE2igZUel.geojson")