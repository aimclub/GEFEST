import matplotlib.pyplot as plt

from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure


class StructVizualizer:
    def __init__(self, domain: Domain):
        self.domain = domain

    def plot_structure(self, struct: Structure, spend_time):
        for poly in struct.polygons:
            self.plot_poly(poly)

        boundary = self.domain.bound_poly
        x = [pt.x for pt in boundary.points]
        y = [pt.y for pt in boundary.points]

        plt.plot(x, y)
        plt.title(str(round(spend_time,2)) + ', sec')
        plt.show()

    def plot_poly(self, poly):
        x = [pt.x for pt in poly.points]
        y = [pt.y for pt in poly.points]
        plt.plot(x, y)
