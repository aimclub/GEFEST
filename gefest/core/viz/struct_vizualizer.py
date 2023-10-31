import matplotlib.pyplot as plt
import moviepy.editor as mp
from matplotlib.lines import Line2D
from moviepy.video.io.bindings import mplfig_to_npimage

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain


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

    def plot_structure(self, structs: list[Structure], domain=None, infos=None, linestyles='-'):
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
        if not isinstance(structs, list):
            structs = [structs]
        if not isinstance(infos, list):
            infos = [infos]
        fig = plt.figure()
        for struct, linestyle in zip(structs, linestyles):
            if self.domain:
                boundary = self.domain.bound_poly
                x = [pt.x for pt in boundary.points]
                y = [pt.y for pt in boundary.points]
                plt.plot(x, y, 'k')
                if domain.prohibited_area:
                    for poly in domain.prohibited_area:
                        self.plot_poly(poly, '-', color='m')

            for poly in struct.polygons:
                self.plot_poly(poly, linestyle)

        lines = [
            Line2D([0], [0], color='black', linewidth=3, linestyle=style) for style in linestyles
        ]
        plt.legend(lines, infos, loc=2)
        return fig

    def plot_poly(self, poly, linestyle, **kwargs):
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
        x_ = [pt.x for pt in poly]
        y_ = [pt.y for pt in poly]

        plt.plot(x_, y_, linestyle=linestyle, **kwargs)
        # for i, p in enumerate(zip(x_, y_)):
        #     plt.plot(p[0], p[1], marker='${}$'.format(i), color='black')


class GIFMaker(StructVizualizer):
    def __init__(self, domain) -> None:
        super().__init__(domain=domain)
        self.frames = []
        self.counter = 0

    def create_frame(self, structure, infos):
        fig = self.plot_structure(structure, self.domain, infos)
        numpy_fig = mplfig_to_npimage(fig)
        self.frames.append(numpy_fig)
        plt.close()

    def make_gif(self, gifname, duration=1500, loop=-1):

        # imgs = [Image.fromarray(img) for img in self.frames]
        # imgs[0].save(f"./{gifname}.fig", save_all=True, append_images=imgs[1:], duration=duration, loop=0)
        # clip = mp.VideoFileClip("mygif.gif")
        clip = mp.ImageSequenceClip(self.frames, durations=[duration] * len(self.frames), fps=1000/duration)
        clip.write_videofile(f'./{gifname}.mp4')
        self.frames = []
        self.counter = 0
