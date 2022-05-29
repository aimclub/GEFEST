import numpy as np
from types import SimpleNamespace

from gefest.core.structure.structure import Structure
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator(domain):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    # Area to length ratio, circle have maximum among all figures (that`s why it`s our optima)
    def area_length_ratio(poly):
        area = domain.geometry.get_square(poly)
        length = domain.geometry.get_length(poly)

        if area == 0:
            return None

        ratio = 1 - 4 * np.pi * area / length ** 2

        return ratio

    # Adding fine for structures containing more (less) than three polygons
    def multi_loss(struct: Structure):
        num = 3
        num_polys = len(struct.polygons)
        loss = 0
        for poly in struct.polygons:
            length = area_length_ratio(poly)
            loss += length
        L = loss + 20 * abs(num_polys - num)

        return L

    estimator = SimpleNamespace()
    estimator.estimate = multi_loss

    # ------------
    # GEFEST estimator
    # ------------

    estimator = Estimator(estimator=estimator)

    return estimator
