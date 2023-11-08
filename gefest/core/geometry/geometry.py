from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from gefest.core.geometry import Point, Polygon


class Geometry(BaseModel, ABC):
    """Abstract geometry class.

    Сlass contains basic transformations of geometries, geometry properties.
    Each of the methods is overridden for a particular dimension of the geometry.
    """

    @abstractmethod
    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        """Resize polygon operation."""
        ...

    @abstractmethod
    def rotate_poly(self, poly: Polygon, angle: float):
        """Rotate polygon operation."""
        ...

    @abstractmethod
    def get_length(self, polygon: Polygon):
        """Perimeter calculation operation."""
        ...

    @abstractmethod
    def get_square(self, polygon: Polygon):
        """Square calculation operation."""
        ...

    @abstractmethod
    def is_contain_point(self, poly: Polygon, point: Point):
        """Checks if point in polygon."""
        ...

    @abstractmethod
    def get_convex(self, poly: Polygon):
        """Returns convex hull."""
        ...

    @abstractmethod
    def intersects(self, poly_1: Polygon, poly_2: Polygon) -> bool:
        """Checks if two polygons intersects."""
        ...

    @abstractmethod
    def min_distance(self, pt_1: Point, pt_2: Point) -> float:
        """Returns min distance between two polygons."""
        ...

    @abstractmethod
    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        """Finds closest point between input point and polygon."""
        ...

    @abstractmethod
    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        """Returns nearest points in the input geometries."""
        ...
