import numpy as np
from types import SimpleNamespace
from uuid import uuid4

from gefest.core.structure.structure import Structure
from gefest.tools.estimators.estimator import Estimator
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.point import Point


def create_internal() -> 'Polygon':
    X = [100, 100, 120, 120, 100]
    Y = [100, 120, 120, 100, 100]

    struct = Polygon(polygon_id=str(uuid4()),
                     points=[(Point(x, y)) for x, y in zip(X, Y)])

    return struct


true = create_internal()
true_struct = Structure([true])


def configurate_estimator(domain):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------

    def multi_loss(struct: Structure):
        if len(struct.polygons) != 1:
            return 1000

        poly_pred = Geometry2D()._poly_to_geom(struct.polygons[0])
        poly_true = Geometry2D()._poly_to_geom(true)

        if not poly_true.intersects(poly_pred):
            return 1000
        if poly_pred.length < poly_true.length / 1.5:
            return 1000

        d1 = poly_pred.difference(poly_true).length
        d2 = poly_true.difference(poly_pred).length

        if d1 < d2:
            return d2
        else:
            return d1

    estimator = SimpleNamespace()
    estimator.estimate = multi_loss

    # ------------
    # GEFEST estimator
    # ------------

    estimator = Estimator(estimator=estimator)

    return estimator
