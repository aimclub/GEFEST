import copy
from typing import List
from core.algs.postproc.resolve_errors import PolygonNotSelfIntersects
from gefest.core.geometry import Point, Polygon, Structure
import numpy as np
from gefest.core.geometry.geometry_2d import Geometry2D


class NoisedPoly():
    """Сlass for revers mutation of synthetic geometry. It is a sampler of geometry,
         that generate simillary to reference polygons.

        class apply a noise for every point of polygon (except first and last for close geometry!!!),
         than it can rotate and resize obtained polygon.
    Args:
         scale (float): - scale factor mapped to domain side. Necessary for obtain
            'sigma' arg for uniform noise function.
                Recommend from 0.01 to 0.05.
                May be tested for other examples;
         init_poly (Polygon): - Reference polygon necessary to generate new reverse mutated geometry;
         resize_scale (List[min,max]): range of min and max value for
            random.uniform generation x_scale and y_scale: (scale value for **x** axis ;scale value for **y** axis).
                Necessary for resize_polygon function;
        degrees_to_rotate (int;float): +/- degrees to rotate polygon;
        rules (List[Rules]): List of geometry validation rules
    """
    geometry = Geometry2D()

    def __init__(self, init_poly:Polygon, scale:float=0.01,
                 resize_scale:List=None, probability:float=0.75,
                 degrees_to_rotate:int=180, rules:List= None,
                 domain=None):
        self.degrees_to_rotate = degrees_to_rotate
        self.proba = probability

        self.init_polygon = init_poly
        self.close = False
        if self.init_polygon[0]==self.init_polygon[-1]:
            self.close=True

        if resize_scale is None:
            resize_scale=[0.75,1.5]
        if rules is None:
            rules = [PolygonNotSelfIntersects()]
        self.rules = rules
        self.x_scale = np.random.uniform(resize_scale[0], resize_scale[1])
        self.y_scale = np.random.uniform(resize_scale[0], resize_scale[1])

        sigma_max_x = max(p.coords[0] for p in self.init_polygon.points)
        sigma_max_y = max(p.coords[1] for p in self.init_polygon.points)
        sigma_min_x = min(p.coords[0] for p in self.init_polygon.points)
        sigma_min_y = min(p.coords[1] for p in self.init_polygon.points)
        max_x = sigma_max_x-sigma_min_x
        max_y = sigma_max_y - sigma_min_y
        self.sigma = np.random.uniform(min(max_x,max_y)*scale,max(max_x,max_y)*scale)

    def __call__(
        self,
        **kwargs,
    ) -> Polygon:
        return self.sample()

    def noising_poly(self,poly: Polygon) -> Polygon:
        x_noise_start = np.random.uniform(-self.sigma, self.sigma)
        y_noise_start = np.random.uniform(-self.sigma, self.sigma)
        for i,point in enumerate(self.init_polygon.points):
            if (i==0 or i==(len(self.init_polygon)-1)) and self.close:
                poly.points.append(Point(point.coords[0] + x_noise_start, point.coords[1] + y_noise_start))
                continue
            if np.random.uniform(0,1) < (1-self.proba):
                poly.points.append(point)
            else:
                x_noise = np.random.uniform(-self.sigma,self.sigma)
                y_noise = np.random.uniform(-self.sigma, self.sigma)
                poly.points.append(Point(point.coords[0] + x_noise, point.coords[1] + y_noise))
        return poly

    def remove_points(self,poly: Polygon) -> Polygon:
        if np.random.uniform(0, 1) < self.proba:
            if len(poly.points)//4 >= 1:
                max_to_del = len(poly.points)//4
            else:
                max_to_del = 1
            for i in range(0,max_to_del):
                pnt_to_del = np.random.randint(1, len(poly.points)-1)
                poly.points.remove(poly.points[pnt_to_del])
        return poly

    def resize_polygon(self,poly: Polygon) -> Polygon:
        """resize polygon size func
        """
        if np.random.uniform(0, 1) < self.proba:
            poly = self.geometry.resize_poly(poly,self.x_scale,self.y_scale)
        return poly

    def rotate_polygon(self,poly: Polygon) -> Polygon:
        angle = np.random.randint(-self.degrees_to_rotate, self.degrees_to_rotate)
        if np.random.uniform(0, 1) < self.proba:
            poly_test = copy.deepcopy(poly)
            poly = self.geometry.rotate_poly(poly, angle)
            if not self.close:
                poly.points.remove(poly.points[0])
                poly_test_2 = copy.deepcopy(poly)
        return poly

    def polygon_transformation(self):
        poly = Polygon([])
        poly = self.noising_poly(poly)
        #poly = self.remove_points(poly)
        #poly = self.resize_polygon(poly)
        poly = self.rotate_polygon(poly)
        return poly

    def sample(self):

        poly = self.polygon_transformation()
        struct = Structure(polygons=([poly]))
        for r in self.rules:
            while not r.validate(struct,0,domain=None):
                poly = self.polygon_transformation()
                struct = Structure(polygons=([poly]))
        return poly