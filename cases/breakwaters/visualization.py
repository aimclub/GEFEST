import matplotlib.pyplot as plt
import numpy as np

from gefest.core.structure.structure import Structure
import cases.breakwaters.configuration.bw_estimator as cost
import cases.breakwaters.configuration.bw_domain as area


def visualize(struct: 'Structure', ax=plt):
    def custom_div_cmap(numcolors=2, name='custom_div_cmap',
                        mincol='black', maxcol='red'):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name=name,
                                                 colors=[mincol, maxcol],
                                                 N=numcolors)
        return cmap

    Z, hs = cost.swan.evaluate(struct)
    polygons = struct.polygons

    poly_area = area.domain.prohibited_area.polygons

    polygons = polygons + poly_area

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
        elif poly.id == 'prohibited_target' or 'prohibited_poly' or 'prohibited_area':
            line_X = [point.x for point in poly.points]
            line_Y = [point.y for point in poly.points]
            ax.plot(line_X,
                    line_Y,
                    color='black',
                    linewidth=4,
                    label='fixed bw')

    custom_map = custom_div_cmap(250, mincol='white', maxcol='black')
    graph = ax.pcolormesh(area.X, area.Y, Z_new, cmap=custom_map, shading='auto', vmax=2.5)

    for target in area.targets:
        ax.scatter(area.X[target[0], target[1]], area.Y[target[0], target[1]], marker='s', s=20, color='green',
                   label='target=' + str(round(hs, 3)))

    ax.axis('off')
    ax.axis(xmin=0, xmax=max(area.coord_X))
    ax.axis(ymin=0, ymax=max(area.coord_Y))
    ax.colorbar(graph)
    ax.legend(fontsize=9)
