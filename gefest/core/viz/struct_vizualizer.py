import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gefest.core.geometry import Structure
from gefest.core.opt.domain import Domain


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

    def plot_structure(self, structs: list[Structure], infos, linestyles="-"):
        """The method displays the given list[obj:`Structure`]
        Args:
            structs: the list[obj:`Structure`] for displaying
            infos: the list of data to plot legend for each structure
            linestyles: pyplot linestyles for stuctures
        Examples:
            >>> from gefest.core.structure.structure import get_random_structure
            >>> struct_1 = get_random_structure(domain)
            >>> struct_2 = get_random_structure(domain)
            >>> viz.plot_structure(
                    [struct_1, struct_2],
                    ['stucture_1', 'stucture_2'],
                    [':', '-'])
        Returns:
            |viz_struct|
        .. |viz_struct| image::https://ibb.co/fN7XCXh
        """

        for struct, linestyle in zip(structs, linestyles):
            for poly in struct.polygons:
                self.plot_poly(poly, linestyle)

            boundary = self.domain.bound_poly
            x = [pt.x for pt in boundary.points]
            y = [pt.y for pt in boundary.points]

            plt.plot(x, y)

        lines = [
            Line2D([0], [0], color="black", linewidth=3, linestyle=style) for style in linestyles
        ]
        plt.legend(lines, infos, loc=2)

    def plot_poly(self, poly, linestyle):
        """The method displays the given :obj:`Polygon`
        Args:
            poly: the :obj:`Polygon` for displaying
            linestyle: pyplot linestyles for polygon
        Examples:
            >>> from gefest.core.structure.structure import get_random_poly
            >>> struct = get_random_structure(domain)
            >>> poly = struct.polygons[0]
            >>> viz.plot_poly(poly, '-')
        Returns:
            .. |viz_poly| image:: https://ibb.co/s31cj3c
        """
        x = [pt.x for pt in poly.points]
        y = [pt.y for pt in poly.points]

        plt.plot(x, y, linestyle=linestyle)
