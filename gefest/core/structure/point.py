from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    """
    The smallest geometrical object which consists spatial information of point

    Args:
        _x (float): the **x** coordinate of Point
        _y (float): the **y** coordinate of Point
        _z (float): the **z** coordinate of Point, by default ``_z=0.0``

    Methods:
        coords: return the list with spatial coordinates of Point

    Returns:
        :class: Point: ``Point(x,y,z)``

    """

    _x: float
    _y: float
    _z: float = 0.0

    # def __init__(self, x: float, y: float, z: float = 0.0):
    #     self._x = x
    #     self._y = y
    #     self._z = z

    @property
    def x(self):
        return round(self._x)

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return round(self._y)

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def z(self):
        return round(self._z)

    @z.setter
    def z(self, value):
        self._z = value

    def coords(self) -> list:
        return [self.x, self.y, self.z]


@dataclass
class Point2D(Point):
    @property
    def z(self):
        return 0

    @z.setter
    def z(self, value):
        self._z = 0

    def coords(self):
        return [self.x, self.y]
