from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    """
    The smallest geometrical object which consists the spatial information of the point

    Args:
        _x (float): the **x** coordinate of Point
        _y (float): the **y** coordinate of Point
        _z (float): the **z** coordinate of Point, by default ``_z=0.0``

    Attributes:
        x (int): returns the rounded until integer **x** coordinate
        y (int): returns the rounded until integer **y** coordinate
        z (int): returns the rounded until integer **z** coordinate

    Returns:
        ``Point(x,y,z)``

    """

    _x: float
    _y: float
    _z: float = 0.0

    @property
    def x(self):
        """int: returns the rounded until integer **x** coordinate
        
        For determinate **x** value or reset it, please use information below:

        Examples:
            >>> point =  Point(1,2,3)
            >>> print([i for i in example_generator(4)])
            [0, 1, 2, 3]
        """
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

    def coords(self) -> list:
        '''returns the list included spatial coordinates of the ``Point``

        Returns:
          ``[x,y,z]``

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
