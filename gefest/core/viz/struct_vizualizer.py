import matplotlib.pyplot as plt

from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure


class StructVizualizer:
    def __init__(self, domain: Domain):
        self.domain = domain

    def plot_structure(self, struct: Structure, info):
        for poly in struct.polygons:
            self.plot_poly(poly, info)

        boundary = self.domain.bound_poly
        x = [pt.x for pt in boundary.points]
        y = [pt.y for pt in boundary.points]

        plt.plot(x, y)
        plt.legend()

    def plot_poly(self, poly, info):
        type = info['type']
        fitness = info['fitness']

        x = [pt.x for pt in poly.points]
        y = [pt.y for pt in poly.points]

        plt.plot(x, y)#, label=f'{type}, fitness = {fitness:.3f}')