from abc import abstractmethod
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon

from typing import List, Union

from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Geometry:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        pass

    @abstractmethod
    def rotate_poly(self, poly: Polygon, angle: float):
        pass

    def get_length(self, polygon: Polygon):
        if len(polygon.points) < 1:
            return 0

        geom_polygon = GeomPolygon([GeomPoint(pt.x, pt.y) for pt in polygon.points])

        return geom_polygon.length

    @abstractmethod
    def get_square(self, polygon: Polygon):
        pass

    @abstractmethod
    def is_contain_point(self, poly: Polygon, point: 'Point'):
        pass

    @abstractmethod
    def get_convex(self, poly: Polygon):
        pass

    @abstractmethod
    def intersects(self, poly_1: Polygon, poly_2: Polygon) -> bool:
        pass

    @abstractmethod
    def distance(self, pt_1: 'Point', pt_2: 'Point') -> float:
        pass
    
    @abstractmethod
    def nearest_point(self, nearest_obj: Union[Point, Polygon], poly: Polygon) -> Point:
        pass
