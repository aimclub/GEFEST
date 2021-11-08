import json
from copy import deepcopy
from dataclasses import dataclass
from random import randint
from typing import List, Optional
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.ops import nearest_points

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


def get_random_structure(min_pols_num=2, max_pols_num=4, min_pol_size=3, max_pol_size=5, domain=None) -> Structure:
    structure = Structure(polygons=[])

    max_pols_num = min(max_pols_num, domain.max_poly_num)
    min_pols_num = min(min_pols_num, domain.max_poly_num)

    num_pols = randint(min_pols_num, max_pols_num)
    is_large = num_pols == 1

    for _ in range(num_pols):
        polygon = get_random_poly(min_pol_size, max_pol_size,
                                  is_large=is_large, parent_structure=structure,
                                  domain=domain)
        if len(polygon.points) > 2:
            structure.polygons.append(polygon)
        else:
            print('Wrong polygon')

    return structure


def get_random_poly(min_pol_size, max_pol_size, is_large: bool,
                    parent_structure: Optional[Structure],
                    domain: Domain) -> Optional[Polygon]:
    geometry = domain.geometry
    try:
        polygon = Polygon(polygon_id=str(uuid4()), points=[])

        polygon.points.extend(deepcopy(domain.fixed_points))

        num_points = randint(min_pol_size, max_pol_size - len(domain.fixed_points))

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
                y_coord = np.random.uniform(low=domain.min_y,
                                            high=domain.max_y)
                x_coord = np.random.uniform(low=domain.min_x,
                                            high=domain.max_x)

                centroid = Point(x_coord, y_coord)
                is_correct_centroid = (domain.contains(centroid) and
                                       all([not geometry.is_contain_point(existing_poly, centroid) for
                                            existing_poly in parent_structure.polygons]))
            if num_iter == 0:
                print('Cannot locate centroid')
                return polygon

        prev_point = centroid
        for _ in range(num_points):
            if is_large:
                point = Point(np.random.uniform(low=domain.min_x, high=domain.max_x),
                              np.random.uniform(low=domain.min_y, high=domain.max_y))
            else:
                if prev_point is not None and not domain.contains(prev_point):
                    raise ValueError('Wrong prev_point')
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

    if prev_point is not None and not geometry.is_contain_point(domain.bound_poly, prev_point):
        print("!!!!!!!!!!!!!!")
        raise ValueError('Wrong prev_point')

    is_correct_point = False
    pt = None
    MAX_ITER = 5000
    num_iter = MAX_ITER
    while not is_correct_point and num_iter > 0:
        try:
            num_iter -= 1
            # print('get rp', MAX_ITER - num_iter)
            pt = Point(
                min(max(np.random.normal(prev_point.x, domain.len_x * 0.05),
                        domain.min_x + domain.len_x * 0.05),
                    domain.max_x - domain.len_x * 0.05),
                min(max(np.random.normal(prev_point.y, domain.len_y * 0.05),
                        domain.min_y + domain.len_y * 0.05),
                    domain.max_y - domain.len_y * 0.05))
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
