import numpy as np
from gefest.core.geometry import Point, Polygon, Structure
import pandas as pd
def poly_from_comsol_txt(path='figures/bottom_square.txt'):
    """

    Args:
        path: path to txt file with comsol points

    Returns:

    """
    res = pd.read_csv(path, sep=' ', header=None)
    points = [[int(round(res.iloc[i, 0], 2)), int(round(res.iloc[i, 1], 2))] for i in res.index]
    points = [Point(i[0], i[1]) for i in np.array(points)]
    poly = Polygon(points=points)
    struct = Structure(polygons=[poly])
    return struct
