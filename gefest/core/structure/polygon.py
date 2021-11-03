from typing import List

from shapely import affinity
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.geometry import Point as GeomPoint

from gefest.core.structure.point import Point


class Polygon:
    def __init__(self, polygon_id: str, points: List[Point]):
        self.polygoin_id = polygon_id
        self.points = points

    # @property
    # def length(self):
    #     return get_length(self)

    # def as_geom(self):
    #     return GeomPolygon([GeomPoint(pt.x, pt.y) for pt in self.points])

    # def plot(self):
    #     x, y = self.as_geom().exterior.xy
    #     plt.plot(x, y)

    # def show(self):
    #     self.plot()
    #     plt.show()

    # def resize(self, x_scale, y_scale):
    #     return resize_poly(self, x_scale, y_scale)
    #
    # def rotate(self, angle: float):
    #     return rotate_poly(self, angle)
    #
    # def contains(self, point: Point):
    #     return is_contain_point(self, point)
