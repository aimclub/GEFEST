from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from gefest.core.geometry import Point, Polygon


class Geometry(BaseModel, ABC):
    """
    Abstract geometry class.
    Сlass contains basic transformations of geometries, geometry properties.
    Each of the methods is overridden for a particular dimension of the geometry.
    """

    @abstractmethod
    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        ...

    @abstractmethod
    def rotate_poly(self, poly: Polygon, angle: float):
        ...

    @abstractmethod
    def get_length(self, polygon: Polygon):
        ...

    @abstractmethod
    def get_square(self, polygon: Polygon):
        ...

    @abstractmethod
    def is_contain_point(self, poly: Polygon, point: Point):
        ...

    @abstractmethod
    def get_convex(self, poly: Polygon):
        ...

    @abstractmethod
    def intersects(self, poly_1: Polygon, poly_2: Polygon) -> bool:
        ...

    @abstractmethod
    def min_distance(self, pt_1: Point, pt_2: Point) -> float:
        ...

    @abstractmethod
    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        ...

    @abstractmethod
    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        ...
