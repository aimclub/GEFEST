import numpy as np
from types import SimpleNamespace

from gefest.core.structure.structure import Structure
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator(domain):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    def len_num_ration(struct: Structure):
        L = []
        p = []
        for poly in struct.polygons:
            L.append(domain.geometry.get_length(poly))
            p.append(len(poly.points))
        length = sum(L)
        num_poly = len(struct.polygons)
        num_points = sum(p)

        ratio = abs(length - 300) + 100 * abs(num_points - 20) + 100 * abs(num_poly - 3)

        return ratio

    estimator = SimpleNamespace()
    estimator.estimate = len_num_ration

    # ------------
    # GEFEST estimator
    # ------------

    estimator = Estimator(estimator=estimator)

    return estimator
