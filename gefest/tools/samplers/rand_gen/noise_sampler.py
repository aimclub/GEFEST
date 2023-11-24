# import copy
from typing import List

import numpy as np

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.postproc.rules import PolygonNotSelfIntersects


class NoisedPoly:
    """Class for revers mutation of synthetic geometry.

    It is a sampler of geometry,
         that generate similar to reference polygons.
        class apply a noise for every point of polygon
        (except first and last for close geometry!!!),
         than it can rotate and resize obtained polygon.
    :param:
         scale (float): - scale factor mapped to domain side. Necessary for obtain
            'sigma' arg for uniform noise function.
                Recommend from 0.01 to 0.05.
                May be tested for other examples;
         init_poly (Polygon): - Reference polygon necessary
          to generate new reverse mutated geometry;
         resize_scale (List[min,max]): range of min and max value for
            random.uniform generation x_scale and y_scale:
             (scale value for **x** axis ;scale value for **y** axis).
                Necessary for resize_polygon function;
        degrees_to_rotate (int;float): +/- degrees to rotate polygon;
        rules (List[Rules]): List of geometry validation rules
    """

    geometry = Geometry2D()

    def __init__(
        self,
        init_poly: Polygon,
        scale: float = 0.01,
        resize_scale: List = None,
        probability: float = 0.75,
        degrees_to_rotate: int = 180,
        rules: List = None,
        is_rotate: bool = True,
        is_resize: bool = True,
        is_remove_points: bool = True,
        domain=None,
    ):
        self.is_rotate = is_rotate
        self.is_resize = is_resize
        self.is_remove_points = is_remove_points

        self.degrees_to_rotate = degrees_to_rotate
        self.proba = probability

        self.init_polygon = init_poly
        self.close = False
        if self.init_polygon[0] == self.init_polygon[-1]:
            self.close = True

        if resize_scale is None:
            self.resize_scale = [0.75, 1.5]
        else:
            self.resize_scale = resize_scale

        if rules is None:
            rules = [PolygonNotSelfIntersects()]

        self.rules = rules

        sigma_max_x = max(p.coords[0] for p in self.init_polygon.points)
        sigma_max_y = max(p.coords[1] for p in self.init_polygon.points)
        sigma_min_x = min(p.coords[0] for p in self.init_polygon.points)
        sigma_min_y = min(p.coords[1] for p in self.init_polygon.points)
        max_x = sigma_max_x - sigma_min_x
        max_y = sigma_max_y - sigma_min_y
        self.sigma = np.random.uniform(min(max_x, max_y) * scale, max(max_x, max_y) * scale)

    def __call__(
        self,
        **kwargs,
    ) -> Polygon:
        """Call method to apply sample() func.

        :param kwargs:
        :return:
        """
        return self.sample()

    def noising_poly(self, poly: Polygon) -> Polygon:
        """Function to add noise to polygon points.

        :param poly:
        :return:
        """
        x_noise_start = np.random.uniform(-self.sigma, self.sigma)
        y_noise_start = np.random.uniform(-self.sigma, self.sigma)
        for i, point in enumerate(self.init_polygon.points):
            if (i == 0 or i == (len(self.init_polygon) - 1)) and self.close:
                poly.points.append(
                    Point(point.coords[0] + x_noise_start, point.coords[1] + y_noise_start)
                )
                continue

            if np.random.uniform(0, 1) < (1 - self.proba):
                poly.points.append(point)

            else:
                x_noise = np.random.uniform(-self.sigma, self.sigma)
                y_noise = np.random.uniform(-self.sigma, self.sigma)
                poly.points.append(Point(point.coords[0] + x_noise, point.coords[1] + y_noise))

        return poly

    def remove_points(self, poly: Polygon) -> Polygon:
        """Function for deleting polygons points.

        :param poly:
        :return:
        """
        if np.random.uniform(0, 1) < self.proba:
            if len(poly.points) // 4 >= 1:
                max_to_del = len(poly.points) // 4
            else:
                max_to_del = 1

            for _ in range(0, max_to_del):  # Choose hwo many points may be deleted
                pnt_to_del = np.random.randint(1, len(poly.points) - 1)
                poly.points.remove(poly.points[pnt_to_del])

        return poly

    def resize_polygon(self, poly: Polygon) -> Polygon:
        """Resize polygon size func.

        :param poly:
        :return:
        """
        x_scale = np.random.uniform(self.resize_scale[0], self.resize_scale[1])
        y_scale = np.random.uniform(self.resize_scale[0], self.resize_scale[1])
        if np.random.uniform(0, 1) < self.proba:
            poly = self.geometry.resize_poly(poly, x_scale, y_scale)

        return poly

    def rotate_polygon(self, poly: Polygon) -> Polygon:
        """Rotating polygon function.

        :param poly:
        :return:
        """
        angle = np.random.randint(-self.degrees_to_rotate, self.degrees_to_rotate)
        if np.random.uniform(0, 1) < self.proba:
            # poly_test = copy.deepcopy(poly)
            poly = self.geometry.rotate_poly(poly, angle)
            if not self.close:
                poly.points.remove(poly.points[0])
                # poly_test_2 = copy.deepcopy(poly)

        return poly

    def polygon_transformation(self):
        """Function applying every notion operation, that may be applied.

        :return:
        """
        poly = Polygon([])
        poly = self.noising_poly(poly)
        if self.is_remove_points:
            poly = self.remove_points(poly)

        if self.is_rotate:
            poly = self.resize_polygon(poly)

        if self.is_resize:
            poly = self.rotate_polygon(poly)

        return poly

    def sample(self):
        """Action function to noise polygon.

        :return: mutated Polygon with validation.
        """
        poly = self.polygon_transformation()
        struct = Structure(polygons=([poly]))
        for r in self.rules:
            while not r.validate(struct, 0, domain=None):
                poly = self.polygon_transformation()
                struct = Structure(polygons=([poly]))

        return poly
