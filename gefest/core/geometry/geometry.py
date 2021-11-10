from abc import abstractmethod
from math import sqrt

from typing import List

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

        total_length = 0
        for i in range(1, len(polygon.points)):
            total_length += sqrt(
                (polygon.points[i - 1].x - polygon.points[i].x) ** 2 +
                (polygon.points[i - 1].y - polygon.points[i].y) ** 2 +
                (polygon.points[i - 1].z - polygon.points[i].z) ** 2)

        total_length += sqrt(
            (polygon.points[len(polygon.points) - 1].x - polygon.points[0].x) ** 2 +
            (polygon.points[len(polygon.points) - 1].y - polygon.points[0].y) ** 2 +
            (polygon.points[len(polygon.points) - 1].z - polygon.points[0].z) ** 2)

        return total_length

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
    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        pass

    @abstractmethod
    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        pass
