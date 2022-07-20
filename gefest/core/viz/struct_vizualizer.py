import matplotlib.pyplot as plt

from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.structure.polygon import Polygon


class StructVizualizer:
    """The object for mapping a :obj:`Structure` or :obj:`Polygon`

    Examples:
        >>> from gefest.core.structure.point import Point
        >>> from gefest.core.structure.polygon import Polygon
        >>> from gefest.core.structure.structure import Structure
        >>> from gefest.core.viz.struct_vizualizer import StructVizualizer
        >>> # creating the rectangle Polygon
        >>> points_rect = [Point(4,0), Point(8,0), Point(8,4), Point(4,4), Point(4,0)]
        >>> rectangle = Polygon('rectangle', points=points_rect)
        >>> # creating the triangle Polygon
        >>> points_triagle = [Point(0,0), Point(3,3), Point(3,0), Point(0,0)]
        >>> triangle = Polygon('triangle', points=points_triagle)
        >>> # creating the Structure and the Vizualizer object
        >>> struct = Structure([triangle, rectangle])
        >>> viz = StructVizualizer()
    """
    def __init__(self, domain: Domain):
        self.domain = domain

    def plot_structure(self, struct: Structure, info: dict) -> plt.plot:
        """_summary_

        Args:
            struct (Structure): _description_
            info (_type_): _description_
        Examples:
            >>> viz.plot_structure

        Returns:
            plt.plot: _description_
        """
        for poly in struct.polygons:
            self.plot_poly(poly, info)

        boundary = self.domain.bound_poly
        x = [pt.x for pt in boundary.points]
        y = [pt.y for pt in boundary.points]

        plt.plot(x, y)
        plt.legend()

    def plot_poly(self, poly: Polygon, info: dict) -> plt.plot:
        type = info['type']
        fitness = info['fitness']

        x = [pt.x for pt in poly.points]
        y = [pt.y for pt in poly.points]

        plt.plot(x, y, label=f'{type}, fitness = {fitness:.3f}')
        plt.legend()
