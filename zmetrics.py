import numpy as np
from gefest.core.geometry.datastructs.structure import Structure

from gefest.core.geometry.domain import Domain
from gefest.tools.estimators.estimator import Estimator
from gefest.core.opt.objective.objective import Objective

from pathlib import Path
import __main__
import inspect
def pth():
    print(inspect.getfile(__main__))

class Area(Objective):
    def __init__(self, domain: Domain, estimator: Estimator = None) -> None:
        super().__init__(domain, estimator)

    def evaluate(self, ind: Structure) -> float:
        area = 0
        for poly in ind:
            area += self.domain.geometry.get_square(poly)
        area = abs(area - (50 * 50))
        norms = []
        if len(ind) == 1:
            for p1, p2 in zip(ind[0][:-1], ind[0][1:]):
                norm = np.linalg.norm(np.array(p1.coords) - np.array(p2.coords))
                norms.append(norm)
        else:
            norms.append(1)
        return area

class SideCoef(Objective):
    def __init__(self, domain: Domain, estimator: Estimator = None) -> None:
        super().__init__(domain, estimator)

    def evaluate(self, ind: Structure) -> float:
        area = 0
        for poly in ind:
            area += self.domain.geometry.get_square(poly)
        area = abs(area - (50 * 50))
        norms = []
        points_num = 0
        if len(ind) == 1:
            for p1, p2 in zip(ind[0][:-1], ind[0][1:]):
                norm = np.linalg.norm(np.array(p1.coords) - np.array(p2.coords))
                norms.append(norm)
            points_num = len(ind[0])
        else:
            norms.append(1)
            points_num = sum(len(p) for p in ind)

        sides_coef = points_num + min(norms) / max(norms)

        return sides_coef
