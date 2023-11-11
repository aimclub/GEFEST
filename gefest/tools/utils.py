import pickle

import numpy as np
import pandas as pd

from gefest.core.geometry import Point, Polygon, Structure


def poly_from_comsol_txt(path: str):
    """Loads txt representation of comsol model.

    Args:
        path: path to txt file with comsol points

    Returns:
        Structure

    """
    res = pd.read_csv(path, sep=' ', header=None)
    points = [[int(round(res.iloc[i, 0], 2)), int(round(res.iloc[i, 1], 2))] for i in res.index]
    points = [Point(i[0], i[1]) for i in np.array(points)]
    poly = Polygon(points=points)
    struct = Structure(polygons=[poly])
    return struct


def load_pickle(path: str):
    """Loads data from pickle."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
        f.close()

    return data
