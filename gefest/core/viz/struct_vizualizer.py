from cProfile import label
import matplotlib.pyplot as plt

from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.structure.polygon import Polygon


class StructVizualizer:
    """The object for mapping a :obj:`Structure` or :obj:`Polygon`

    Examples:
        >>> from gefest.core.structure.domain import Domain
        >>> from gefest.core.viz.struct_vizualizer import StructVizualizer
        >>> domain = Domain()
        >>> viz = StructVizualizer(domain)
    """

    def __init__(self, domain: Domain):
        self.domain = domain

    def plot_structure(self, struct: Structure) -> plt.plot:
        """The method displays the given :obj:`Structure`

        Args:
            struct (Structure): the :obj:`Structure` for displaying

        Examples:
            >>> from gefest.core.structure.structure import get_random_structure
            >>> struct = get_random_structure(domain)
            >>> viz.plot_structure(struct)

        Returns:
            plt.plot: |viz_struct|

        .. |viz_struct| image:: https://i.ibb.co/r0YsVtR/vizualizer.png
        """

        for poly in struct.polygons:
            self.plot_poly(poly, poly.id)

        boundary = self.domain.bound_poly
        x = [pt.x for pt in boundary.points]
        y = [pt.y for pt in boundary.points]

        plt.plot(x, y, label='allowed area')
        plt.legend()

    def plot_poly(self, poly: Polygon, info: str) -> plt.plot:
        """The method displays the given :obj:`Polygon`

        Args:
            poly (Polygon): the :obj:`Polygon` for displaying
            info (str): name of Polygon, allow to use id of Polygon

        Examples:
            >>> from gefest.core.structure.structure import get_random_poly
            >>> struct = get_random_structure(domain)
            >>> poly = struct.polygons[0]
            >>> viz.plot_poly(poly, 'random genreated polygon')

        Returns:
            plt.plot: |viz_poly|

        .. |viz_poly| image:: https://i.ibb.co/x7B0QPY/random-poly.png
        """
        x = [pt.x for pt in poly.points]
        y = [pt.y for pt in poly.points]

        plt.plot(x, y, label=info)
        plt.legend()
