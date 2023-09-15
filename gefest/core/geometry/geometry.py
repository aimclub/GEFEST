from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel
from shapely.geometry import LineString
from shapely.geometry import Point as GeomPoint

from gefest.core.geometry import Point, Polygon


class Geometry(BaseModel, ABC):
    """
    Abstract geometry class.
    Сlass contains basic transformations of geometries, geometry properties.
    Each of the methods is overridden for a particular dimension of the geometry.
    """

    @abstractmethod
    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        pass

    @abstractmethod
    def rotate_poly(self, poly: Polygon, angle: float):
        pass

    @abstractmethod
    def get_length(self, polygon: Polygon):
        pass

    @abstractmethod
    def get_square(self, polygon: Polygon):
        pass

    @abstractmethod
    def is_contain_point(self, poly: Polygon, point: Point):
        pass

    @abstractmethod
    def get_convex(self, poly: Polygon):
        pass

    @abstractmethod
    def intersects(self, poly_1: Polygon, poly_2: Polygon) -> bool:
        pass

    @abstractmethod
    def min_distance(self, pt_1: Point, pt_2: Point) -> float:
        pass

    @abstractmethod
    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        pass

    @abstractmethod
    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        pass
