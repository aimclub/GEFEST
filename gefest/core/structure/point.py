from dataclasses import dataclass


@dataclass
class Point:
    """
    The smallest object in GEFEST

    Args:
        x(float): set the **x** coordinate of Point()
        y(float): set the **y** coordinate of Point()
        z(float): set the **z** coordinate of Point(), by default z = 0.0

    Returns:
        Point: Point(x,y,z)
    """

    _x: float
    _y: float
    _z: float = 0.0

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

    def coords(self):
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
