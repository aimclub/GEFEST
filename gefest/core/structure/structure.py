import json
from copy import deepcopy
from dataclasses import dataclass
from random import randint
from typing import List, Optional
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np

from gefest.core.algs.geom.validation import MIN_DIST, self_intersection
from gefest.core.structure.domain import Domain
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


@dataclass
class Structure:
    polygons: List[Polygon]

    def __str__(self):
        out_str = ''
        for i, pol in enumerate(self.polygons):
            out_str += f'\r\n Polygon {i}, size {len(pol.points)}: \r\n'
            for j, pt in enumerate(pol.points):
                out_str += f'Point {j}: x={round(pt.x, 2)}, y={round(pt.y, 2)}; '
        return out_str

    def __repr__(self):
        return json.dumps(self, default=vars)

    @property
    def text_id(self):
        out_str = ''
        for i, pol in enumerate(self.polygons):
            out_str += f'P{i}={len(pol.points)}:'
            for j, pt in enumerate(pol.points):
                out_str += f'(x={round(pt.x, 2)}, y={round(pt.y, 2)}); '
        return out_str

    @property
    def length(self):
        return sum([p.length for p in self.polygons])

    @property
    def size(self):
        return sum([len(p.points) for p in self.polygons])

    def plot(self, structure, domain=None, title=None):
        x = [point._x for point in structure.polygons[0].points]
        y = [point._y for point in structure.polygons[0].points]
        plt.plot(x, y)
        plt.title(title)
        plt.show()


def get_random_structure(domain) -> Structure:
    structure = Structure(polygons=[])

    num_pols = randint(domain.min_poly_num, domain.max_poly_num)
    is_large = num_pols == 1

    for _ in range(num_pols):
        polygon = get_random_poly(is_large=is_large,
                                  parent_structure=structure,
                                  domain=domain)
        if polygon is not None and len(polygon.points) > 2:
            structure.polygons.append(polygon)
        else:
            print('Wrong polygon')

    return structure


def get_random_poly(is_large: bool,
                    parent_structure: Optional[Structure],
                    domain: Domain) -> Optional[Polygon]:
    geometry = domain.geometry
    try:
        polygon = Polygon(polygon_id=str(uuid4()), points=[])

        polygon.points.extend(deepcopy(domain.fixed_points))

        num_points = randint(domain.min_points_num, domain.max_points_num)
        # default centroid
        centroid = Point(np.random.uniform(low=domain.min_x, high=domain.max_x),
                         np.random.uniform(low=domain.min_y, high=domain.max_y))

        # set centroids
        if parent_structure is not None:
            # more correct placements of centroids
            is_correct_centroid = False
            num_iter = 5000
            while not is_correct_centroid and num_iter > 0:
                num_iter -= 1

                centroid = Point(np.random.uniform(low=domain.min_x, high=domain.max_x),
                                 np.random.uniform(low=domain.min_y, high=domain.max_y))

                is_correct_centroid = (all([not geometry.is_contain_point(existing_poly, centroid) for
                                            existing_poly in parent_structure.polygons]))

            if num_iter == 0:
                print('Cannot locate centroid')
                return polygon

        prev_point = centroid
        for _ in range(num_points):
            if is_large:
                point = create_next_point(prev_point, domain)
                prev_point = point
            else:
                point = get_random_point(prev_point, domain=domain)

                if parent_structure is not None:
                    is_correct_point = False
                    iter_num = 100
                    while not is_correct_point and iter_num > 0:
                        iter_num -= 1

                        if (prev_point is not None and
                                not geometry.is_contain_point(domain.bound_poly, prev_point)):
                            raise ValueError('Wrong prev_point')

                        point = get_random_point(prev_point, polygon,
                                                 parent_structure=parent_structure,
                                                 domain=domain)
                        if point is None:
                            iter_num = 0
                            continue

                        is_correct_point = \
                            all([not geometry.is_contain_point(existing_poly, point)
                                 for existing_poly in parent_structure.polygons]) \
                            and not self_intersection(Structure([Polygon('tmp', polygon.points + [point])]))

                    if iter_num == 0:
                        print('Preliminary return of poly')
                        return polygon

                prev_point = point

            polygon.points.append(point)

    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return None
    return polygon


def get_random_point(prev_point: Point,
                     parent_poly: Optional[Polygon] = None,
                     parent_structure: Optional[Structure] = None,
                     domain=None) -> Optional[Point]:
    geometry = domain.geometry

    is_correct_point = False
    pt = None
    MAX_ITER = 100
    num_iter = MAX_ITER
    while not is_correct_point and num_iter > 0:
        try:
            num_iter -= 1
            pt = create_next_point(prev_point, domain)
            is_correct_point = domain.contains(pt)

            if (is_correct_point and parent_poly and
                    len(parent_poly.points) > 0 and num_iter > MAX_ITER / 2):
                is_correct_point = all([geometry.distance(pt, poly_pt) > domain.len_x * 0.1
                                        for poly_pt in parent_poly.points])

            if is_correct_point and parent_structure and len(parent_structure.polygons) > 0:
                # check then new point is not near existing polygons
                for poly_from_structure in parent_structure.polygons:
                    if geometry.get_length(poly_from_structure) != geometry.get_length(parent_poly):
                        # TODO more smart check
                        nearest_pt = geometry.nearest_point(pt, poly_from_structure)
                        is_correct_point = geometry.distance(pt, nearest_pt) > MIN_DIST
                        if not is_correct_point:
                            break
        except Exception as ex:
            import traceback
            print(traceback.format_exc())
            print(ex)

    if num_iter == 0:
        print('Preliminary return of point')
        return None
    return pt


def create_next_point(prev_point: Point, domain) -> Point:
    pt = Point(
        min(max(np.random.normal(prev_point.x, domain.len_x * 0.05),
                domain.min_x + domain.len_x * 0.05),
            domain.max_x - domain.len_x * 0.05),
        min(max(np.random.normal(prev_point.y, domain.len_y * 0.05),
                domain.min_y + domain.len_y * 0.05),
            domain.max_y - domain.len_y * 0.05))
    return pt
