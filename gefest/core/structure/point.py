from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    """
    The smallest geometrical object which consists the spatial information of the point

    Args:
        _x: the **x** coordinate of :obj:`Point`
        _y: the **y** coordinate of :obj:`Point`
        _z: the **z** coordinate of :obj:`Point`, by default ``_z=0.0``

    Attributes:
        x: returns the rounded until integer **x** coordinate
        y: returns the rounded until integer **y** coordinate
        z: returns the rounded until integer **z** coordinate

    Examples:
            >>> from gefest.core.structure.point import Point
            >>> point = Point(1,2)
            >>> point.coords()
            [1, 2, 0]

            >>> print('x = {}, y = {}'.format(point.x, point.y))
            x = 1, y = 2

    Returns:
        Point: ``Point(x,y,z)``

    """

    _x: float
    _y: float
    _z: float = 0.0

    @property
    def x(self) -> int:
        return round(self._x)

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self) -> int:
        return round(self._y)

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def z(self) -> int:
        return round(self._z)

    @z.setter
    def z(self, value):
        self._z = value

    def coords(self):
        '''Returns the :obj:`list` included spatial coordinates of the :obj:`Point`

        Returns:
          :obj:`List`: ``[x,y,z]``

        '''
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
