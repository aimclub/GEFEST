from typing import List, Optional, Tuple

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Domain:
    """:obj:`Domain` is responsible for the whole information about geometry of the
    problem

    Args:
        name (str): the name 'id' of the :obj:`Domain`, by deafult ``name='main'``
        allowed_area (List[Tuple]): determinate allowed area for exploring solution
            into its frame; the :obj:`list` of :obj:`tuple` objects that
            contain couples of border coordinates, by default is ``None``
            If ``allowed_area=None``, allowed area will be determinated by square
            with the ``length of edge = 100`` and bottom left corner located in the origin.
        max_poly_num (int): the maximum number of :obj:`Polygon` objects :obj:`Structure`
            might contains, by default ``max_poly_num=4``
        min_poly_num (int): the minimum number of :obj:`Polygon` objects :obj:`Structure`
            might contains, by default ``min_poly_num=2``
        max_points_num (int): the maximum number of :obj:`Point` objects :obj:`Polygon`
            might contains, by default ``max_points_num=50``
        min_points_num (int): the minimum number of :obj:`Point` objects :obj:`Polygon`
            might contains, by default ``min_points_num=20``
        fixed_points (list): determine the areas that must not be ignored during find solution;
            the :obj:`list` of sets of border coordinates, every set has contain couples of coordinates
            as set of :obj:`tuple`, by default is ``None``
        is_closed (bool): will create geometrical objects with closed borders (when start point is same
            with the last one) if ``True``, against if ``False``; by default is ``True``
        geometry (obj): determinate a way for processing created objects, by default is ``None``
            If ``geometry=None``, created objects will process as 2D objects via :obj:`Geometry2D()`

    Attributes:
        min_x (int): the minimum value among **x** coordinates within **allowed_area**
        max_x (int): the maximum value among **x** coordinates within **allowed_area**
        min_y (int): the minimum value among **y** coordinates within **allowed_area**
        max_y (int): the maximum value among **y** coordinates within **allowed_area**
        len_x (int): the absolute difference betwen **max_x** and **min_x**
        len_y (int): the absolute difference betwen **max_y** and **min_y**
        bound_poly (Polygon): creates the :obj:`Polygon` by :obj:`Domain`'s border coordinates

    Returns:
        Domain: ``obj Domain()``

    """
    def __init__(self, name='main', allowed_area: Optional[List[Tuple]] = None,
                 max_poly_num=4, min_poly_num=2,
                 max_points_num=50, min_points_num=20,
                 fixed_points=None,
                 is_closed=True,
                 geometry=None):
        self.name = name
        self.is_closed = is_closed
        if geometry is None:
            self.geometry = Geometry2D()
        else:
            self.geometry = geometry

        if allowed_area is None:
            allowed_area = [(0, 0),
                            (0, 100),
                            (100, 100),
                            (100, 0)]
        else:
            allowed_area = [(int(round(p[0], 0)), int(round(p[1], 0))) for p in allowed_area]

        self.allowed_area = allowed_area

        self.max_poly_num = max_poly_num
        self.min_poly_num = min_poly_num

        self.max_points_num = max_points_num
        self.min_points_num = min_points_num

        self.min_dist = max(self.max_x - self.min_x, self.max_y - self.min_y) / 35

        self.fixed_points = [Polygon(polygon_id='fixed', points=[Point(p[0], p[1]) for p in poly]) for poly in
                             fixed_points] \
            if fixed_points is not None else []

    @property
    def min_x(self):
        return min(p[0] for p in self.allowed_area)

    @property
    def max_x(self):
        return max(p[0] for p in self.allowed_area)

    @property
    def min_y(self):
        return min(p[1] for p in self.allowed_area)

    @property
    def max_y(self):
        return max(p[1] for p in self.allowed_area)

    @property
    def len_x(self):
        return abs(self.max_x - self.min_x)

    @property
    def len_y(self):
        return abs(self.max_y - self.min_y)

    def contains(self, point: Point):
        '''returns ``True`` if given :obj:`Point` locates in the allowed area borders,
        otherwise returns ``False``

        Returns:
            :obj:`bool`: ``True``/``False``
        '''
        geom_poly_allowed = Polygon(polygon_id=f'bnd_{self.name}',
                                    points=[Point(pt[0], pt[1]) for pt in self.allowed_area])
        return self.geometry.is_contain_point(geom_poly_allowed, point)

    @property
    def bound_poly(self):
        if self.allowed_area is None or len(self.allowed_area) <= 2:
            raise ValueError('Not enough points for domain')
        bnd_points = [Point(*pt_coords) for pt_coords in self.allowed_area]
        return Polygon(polygon_id=f'bnd_{self.name}', points=bnd_points)
