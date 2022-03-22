import timeit

import matplotlib.pyplot as plt
import numpy as np

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.simulators.swan.swan_interface import Swan
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.opt.analytics import EvoAnalytics

"""
Here is an example of breakwaters optimization. SWAN model need to be installed.
You can find our configuration in simulators folder in INPUT file.
It consist water area with two fixed breakwaters, bathymetry (specified in bathymetry folder) and land.
Output file (wave height at each point of the water are) located is in the 'r' folder.
"""

"""
Below are some settings for domain to be researched.
You have to set grid resolution on each axis (x and y), spatial grid
and coordinates of your target (or targets) for which you want to optimize height of wave.
"""
grid_resolution_x = 83  # Number of points on x-axis
grid_resolution_y = 58  # Number of points on y-axis
coord_X = np.linspace(0, 2075, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(0, 1450, grid_resolution_y + 1)  # Y coordinate for spatial grid
X, Y = np.meshgrid(coord_X, coord_Y)  # Two dimensional spatial grid
grid_target_X = 25  # X-grid coordinate of your target
grid_target_Y = 25  # Y-grid coordinate of your target

"""
Create domain grid and coordinates of your targets.
As you can see, in this exampe we consider only one target
"""
grid = [grid_resolution_x, grid_resolution_y]
targets = [[grid_target_X, grid_target_Y]]

"""
Below we have domain configuration that GEFEST requires for every task.
Here we are working with open polygons
"""
fixed_points = [[[1000, 50], [700, 600], [800, 800]], [[1900, 540], [1750, 1000]]]
is_closed = False
geometry = Geometry2D(is_closed=is_closed)
domain = Domain(allowed_area=[(min(coord_X), min(coord_Y)),
                              (min(coord_X), max(coord_Y)),
                              (max(coord_X), max(coord_Y)),
                              (max(coord_X), min(coord_Y))],
                geometry=geometry,
                max_poly_num=1,
                min_poly_num=1,
                max_points_num=10,
                min_points_num=2,
                fixed_points=fixed_points,
                is_closed=is_closed)
task_setup = Setup(domain=domain)

"""
Here is the preparation of the SWAN model
You need to set path to folder with swan.exe file
Our SWAN interface uses this path, domain grid, GEFEST domain and coordinates of targets
"""
path = '../../gefest/simulators/swan/swan_model/'
swan = Swan(path=path,
            targets=targets,
            grid=grid,
            domain=domain)

max_length = np.linalg.norm(np.array([max(coord_X) - min(coord_X), max(coord_Y) - min(coord_Y)]))


# Cost function defining as sum of cost of structure and wave height at the target points
def cost(struct: Structure):
    lengths = 0
    for poly in struct.polygons:
        if poly.id != 'fixed':
            length = geometry.get_length(poly)
            lengths += length

    Z, hs = swan.evaluate(struct)
    loss = lengths / max_length + hs

    return loss


# Optimizing
start = timeit.default_timer()
optimized_structure = optimize(task_setup=task_setup,
                               objective_function=cost,
                               pop_size=10,
                               max_gens=10)
spend_time = timeit.default_timer() - start


# Some visualization for structures and domain
def visualize(struct: 'Structure', ax=plt):
    def custom_div_cmap(numcolors=2, name='custom_div_cmap',
                        mincol='black', maxcol='red'):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name=name,
                                                 colors=[mincol, maxcol],
                                                 N=numcolors)
        return cmap

    Z, hs = swan.evaluate(struct)
    polygons = struct.polygons

    Z_new = []
    for z in Z:
        z_new = []
        for k in z:
            if k <= 0:
                z_new.append(0)
            else:
                z_new.append(k)
        Z_new.append(z_new)
    Z_new = np.array(Z_new)

    for poly in polygons:
        if poly.id == 'tmp':
            line_X = [point.x for point in poly.points]
            line_Y = [point.y for point in poly.points]
            ax.plot(line_X,
                    line_Y,
                    color='red',
                    linewidth=2,
                    label='breakwater',
                    marker='o')
        elif poly.id == 'fixed':
            line_X = [point.x for point in poly.points]
            line_Y = [point.y for point in poly.points]
            ax.plot(line_X,
                    line_Y,
                    color='black',
                    linewidth=4,
                    label='fixed bw')

    custom_map = custom_div_cmap(250, mincol='white', maxcol='black')
    ax.pcolormesh(X, Y, Z_new, cmap=custom_map, shading='auto')

    for target in targets:
        ax.scatter(X[target[0], target[1]], Y[target[0], target[1]], marker='s', s=20, color='green',
                   label='target=' + str(round(hs, 3)))

    ax.axis('off')
    ax.axis(xmin=0, xmax=max(coord_X))
    ax.axis(ymin=0, ymax=max(coord_Y))
    ax.legend(fontsize=9)


visualize(optimized_structure)
plt.show()
EvoAnalytics.create_boxplot()
