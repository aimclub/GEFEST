import json
import pickle
from uuid import uuid4
from dataclasses import dataclass
from random import randint, choice, random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from gefest.core.structure.domain import Domain
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


@dataclass
class Structure:
    """The geometrical object made up of :obj:`Polygon` objects
    Args:
        polygons: list of :obj:`Polygon` objects which form a combined set of polygons,
            needed for joint processing capability of polygons
    Attributes:
        text_id: returns information about :obj:`Polygons` and :obj:`Points`
            included in :obj:`Structure`
        polygons: returns the :obj:`list` of :obj:`Polygon` objects
        total_points: returns the :obj:`list` with lengths (number of :obj:`Point`)
            of every :obj:`Polygon` included
    Examples:
        >>> from gefest.core.structure.point import Point
        >>> from gefest.core.structure.polygon import Polygon
        >>> from gefest.core.structure.structure import Structure
        >>> # creating the rectangle Polygon
        >>> points_rect = [Point(4,0), Point(8,0), Point(8,4), Point(4,4), Point(4,0)]
        >>> rectangle = Polygon('rectangle', points=points_rect)
        >>> # creating the triangle Polygon
        >>> points_triagle = [Point(0,0), Point(3,3), Point(3,0), Point(0,0)]
        >>> triangle = Polygon('triangle', points=points_triagle)
        >>> # creating the Structure and plot it
        >>> struct = Structure([triangle, rectangle])
        >>> struct.text_id
        'P0=4:(x=0, y=0); (x=3, y=3); (x=3, y=0); (x=0, y=0);
        P1=5:(x=4, y=0); (x=8, y=0); (x=8, y=4); (x=4, y=4); (x=4, y=0); '
        >>> struct.total_points
        [4, 5]
    Returns:
        Structure: ``Structure(List[Polygon])``
    """

    polygons: List[Polygon]

    def __str__(self):
        out_str = ""
        for i, pol in enumerate(self.polygons):
            out_str += f"\r\n Polygon {i}, size {len(pol.points)}: \r\n"
            for j, pt in enumerate(pol.points):
                out_str += f"Point {j}: x={round(pt.x, 2)}, y={round(pt.y, 2)}; "
        return out_str

    def __repr__(self):
        return json.dumps(self, default=vars)

    @property
    def text_id(self):
        out_str = ""
        for i, pol in enumerate(self.polygons):
            out_str += f"P{i}={len(pol.points)}:"
            for j, pt in enumerate(pol.points):
                out_str += f"(x={round(pt.x, 2)}, y={round(pt.y, 2)}); "
        return out_str

    @property
    def length(self):
        return sum([p.length for p in self.polygons])

    def total_points(self) -> list:
        return [len(p.points) for p in self.polygons]

    @property
    def size(self):
        return sum([len(p.points) for p in self.polygons])

    def plot(
        self,
        title=None,
        legend=False,
        color=None,
        grid=False,
        ax=None,
        show_coords=False,
    ):
        """Visualization with drawn :obj:`Strucrure`

        Args:
            title: the name of drawing, by default ``None``
            legend: show legend (id of polygons), by default ``False``
            color: set special color for all polygons, see variants of colors in `matplotlib documentation`_
            ax: set name of axes where drown structure has to be part of complicated matplotlib figure,
             by default `None`
            show_coords: print coords for every apex of polygon(s) if `True`, by default `False`

        Examples:
            >>> struct.plot(legend=True)

        Returns:
            plot: |viz|

        .. |viz| image:: https://i.ibb.co/1q0CVNJ/structure-plot.png
        .. _matplotlib documentation:
            https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
        """

        for poly in self.polygons:
            try:
                poly_id = poly.id
                poly_id = poly_id.split("-")[-1]
            except IndexError:
                poly_id = poly.id

            if ax:
                x = [point._x for point in poly.points]
                y = [point._y for point in poly.points]

                ax.plot(x, y, label=poly.id, c=color, marker="o")
                if show_coords:
                    ax.scatter(x, y, marker="o", c=color)
                    text_x = [str(round(x, 1)) for x in x]
                    text_y = [str(round(y, 1)) for y in y]
                    for idx, text in enumerate(zip(text_x, text_y)):
                        ax.annotate(
                            str(text),
                            (x[idx] + 0.01, y[idx] + 0.01),
                            rotation=45.0,
                            fontsize=8,
                        )
            else:
                x = [point._x for point in poly.points]
                y = [point._y for point in poly.points]
                plt.plot(x, y, label=poly.id, c=color, marker="o")
                if show_coords:
                    plt.scatter(x, y, marker="o", c=color)
                    text_x = [str(round(x, 1)) for x in x]
                    text_y = [str(round(y, 1)) for y in y]
                    for idx, text in enumerate(zip(text_x, text_y)):
                        plt.annotate(
                            str(text),
                            (x[idx] + 0.01, y[idx] + 0.01),
                            rotation=45.0,
                            fontsize=8,
                        )
        if legend:
            plt.legend()
        if grid:
            plt.grid()
        plt.title(title)


def get_random_structure(domain) -> Structure:
    # Creating structure with random number of polygons

    structure = Structure(polygons=[])

    num_pols = randint(domain.min_poly_num, domain.max_poly_num)

    for _ in range(num_pols):
        polygon = get_random_poly(parent_structure=structure, domain=domain)
        if polygon is not None and len(polygon.points) > 1:
            structure.polygons.append(polygon)
        else:
            continue

    return structure


def get_random_poly(
    parent_structure: Optional[Structure], domain: Domain
) -> Optional[Polygon]:
    geometry = domain.geometry
    try:
        """
        Function for creation random polygon.
        The main idea is to create centroids along with a neighborhood to locate the polygon.
        Neighborhood sizes range from small to large.
        The main condition for creating a neighborhood is the absence of other polygons in it.
        After setting the neighborhood, polygons are created around the centroid inside the given neighborhood.
        This approach is less demanding on postprocessing than random creation
        """

        # Centroid with it neighborhood called occupied area
        occupied_area = create_area(domain, parent_structure, geometry)
        if occupied_area is None:
            # If it was not possible to find the occupied area then returns None
            return None
        else:
            centroid = occupied_area[0]
            sigma = occupied_area[1]  # Size of neighborhood
            # The polygon is created relative to the centroid
            # and the size of the neighborhood
            polygon = create_poly(centroid, sigma, domain, geometry)
    except Exception as ex:
        print(ex)
        import traceback

        print(traceback.format_exc())
        return None

    return polygon


def get_random_point(
    polygon: "Polygon", structure: "Structure", domain: "Domain"
) -> Optional[Point]:
    # Creating a point to fill the polygon

    centroid = domain.geometry.get_centroid(polygon)
    sigma = distance(centroid, structure, domain.geometry) / 3
    point = create_polygon_point(centroid, sigma)
    max_attempts = 20  # Number of attempts to create in bound point
    while not in_bound(point, domain):
        point = create_polygon_point(centroid, sigma)
        max_attempts -= 1
        if max_attempts == 0:
            return None
    return point


def create_poly(
    centroid: "Point", sigma: int, domain: "Domain", geometry: "Geometry"
) -> Polygon:
    # Creating polygon in the neighborhood of the centroid
    # sigma defines neighborhood

    num_points = randint(
        domain.min_points_num, domain.max_points_num
    )  # Number of points in a polygon
    points = []
    for _ in range(num_points):
        point = create_polygon_point(centroid, sigma)  # point in polygon
        while not in_bound(point, domain):  # checking if a point is in domain
            point = create_polygon_point(centroid, sigma)
        points.append(point)
    if domain.is_closed:
        points.append(points[0])

    poly = geometry.get_convex(
        Polygon(polygon_id=str(uuid4()), points=points)
    )  # avoid self intersection in polygon

    return poly


def create_area(domain: "Domain", structure: "Structure", geometry: "Geometry"):
    n_poly = len(structure.polygons)  # Number of already existing polygons
    area_size = np.random.randint(low=3, high=15)  # Neighborhood compression ratio
    sigma = (
        max(domain.max_x - domain.min_x, domain.max_y - domain.min_y) / area_size
    )  # Neighborhood size
    if n_poly == 0:
        # In the absence of polygons, the centroid can be located anywhere
        centroid = create_random_point(domain)
    else:
        """
        This procedure allows to find a centroid in the neighborhood
        of which there are no other polygons.
        The minimum distance must be less than 2.5 * sigma.
        """
        centroid = create_random_point(domain)
        min_dist = distance(
            centroid, structure, geometry
        )  # Distance to the nearest polygon in the structure
        max_attempts = 20
        while min_dist < 2.5 * sigma:
            area_size = np.random.randint(low=3, high=15)
            sigma = (
                max(domain.max_x - domain.min_x, domain.max_y - domain.min_y)
                / area_size
            )
            centroid = create_random_point(domain)
            min_dist = distance(centroid, structure, geometry)
            if max_attempts == 0:
                return None
            max_attempts -= 1

    return centroid, sigma


def create_random_point(domain: "Domain") -> Point:
    point = Point(
        np.random.uniform(low=domain.min_x, high=domain.max_x),
        np.random.uniform(low=domain.min_y, high=domain.max_y),
    )
    while not in_bound(point, domain):
        point = Point(
            np.random.uniform(low=domain.min_x, high=domain.max_x),
            np.random.uniform(low=domain.min_y, high=domain.max_y),
        )

    return point


def create_polygon_point(centroid: "Point", sigma: int) -> Point:
    # Creating polygon point inside the neighborhood defined by the centroid
    point = Point(
        np.random.normal(centroid.x, sigma, 1)[0],
        np.random.normal(centroid.y, sigma, 1)[0],
    )

    return point


def in_bound(point: "Point", domain: "Domain") -> bool:
    poly_domain = Polygon(points=[Point(c[0], c[1]) for c in domain.allowed_area])
    return domain.geometry.is_contain_point(poly_domain, point)


def distance(point: "Point", structure: "Structure", geometry: "Geometry") -> float:
    polygons = structure.polygons
    distances = []
    for poly in polygons:
        d = geometry.centroid_distance(point, poly)
        distances.append(d)

    return min(distances)


def get_structure_from_path(path: str):
    """Allows to get structure from ``.pkl``

    Args:
        path (str): path to ``.pkl`` file contains ``Structure``

    Returns:
        Structure: structure from file
    """

    open_file = open(path, "rb")
    structure = pickle.load(open_file)
    open_file.close()

    return structure[0]
