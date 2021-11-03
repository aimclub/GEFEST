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

from gefest.core.structure.geometry import MIN_DIST, self_intersection
from gefest.core.structure.polygon import Polygon, PolygonPoint
from gefest.core.utils import GlobalEnv
from shapely import affinity

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

    def plot(self, domain=None, title=None):
        if domain is None:
            domain = GlobalEnv().domain

        for poly in self.polygons:
            poly.plot()
        if isinstance(domain, list):
            for d in domain:
                geom_poly_allowed = GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in d.allowed_area])
                x, y = geom_poly_allowed.exterior.xy
                plt.plot(x, y)
        else:
            geom_poly_allowed = GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in domain.allowed_area])
            x, y = geom_poly_allowed.exterior.xy
            plt.plot(x, y)
        if title:
            plt.title(title)
        plt.show()


def get_random_structure(min_pols_num=2, max_pols_num=4, min_pol_size=3, max_pol_size=5, domain=None) -> Structure:
    structure = Structure(polygons=[])
    if domain is None:
        current_domain = GlobalEnv().domain
    else:
        current_domain = domain

    max_pols_num = min(max_pols_num, domain.max_poly_num)
    min_pols_num = min(min_pols_num, domain.max_poly_num)

    num_pols = randint(min_pols_num, max_pols_num)
    is_large = num_pols == 1

    for _ in range(num_pols):
        polygon = get_random_poly(min_pol_size, max_pol_size,
                                  is_large=is_large, parent_structure=structure,
                                  domain=current_domain)
        if len(polygon.points) > 2:
            structure.polygons.append(polygon)
        else:
            print('Wrong polygon')

    return structure


def get_random_poly(min_pol_size=3, max_pol_size=5, is_large=False,
                    parent_structure: Optional[Structure] = None,
                    domain=None) -> Optional[Polygon]:
    try:
        if domain is None:
            current_domain = GlobalEnv().domain
        else:
            current_domain = domain
        polygon = Polygon(polygon_id=str(uuid4()), points=[])

        polygon.points.extend(deepcopy(current_domain.fixed_points))

        num_points = randint(min_pol_size, max_pol_size - len(current_domain.fixed_points))

        # default centroid
        centroid = PolygonPoint(np.random.uniform(low=current_domain.min_x, high=current_domain.max_x),
                                np.random.uniform(low=current_domain.min_y, high=current_domain.max_y))

        # set centroids
        if parent_structure is not None:
            # more correct placements of centroids
            is_correct_centroid = False
            num_iter = 5000
            while not is_correct_centroid and num_iter > 0:
                num_iter -= 1
                y_coord = np.random.uniform(low=current_domain.min_y,
                                            high=current_domain.max_y)
                x_coord = np.random.uniform(low=current_domain.min_x,
                                            high=current_domain.max_x)

                centroid = PolygonPoint(x_coord, y_coord)
                is_correct_centroid = (current_domain.contains(centroid) and
                                       all([not existing_poly.contains(centroid) for
                                            existing_poly in parent_structure.polygons]))
            if num_iter == 0:
                print('Cannot locate centroid')
                return polygon

        prev_point = centroid
        for _ in range(num_points):
            if is_large:
                point = PolygonPoint(np.random.uniform(low=current_domain.min_x, high=current_domain.max_x),
                                     np.random.uniform(low=current_domain.min_y, high=current_domain.max_y))
            else:
                if prev_point is not None and not current_domain.contains(prev_point):
                    print("!!!!!!!!!!!!!!3")
                    raise ValueError('Wrong prev_point')
                point = get_random_point(prev_point, domain=current_domain)

                if parent_structure is not None:
                    is_correct_point = False
                    iter_num = 100
                    while not is_correct_point and iter_num > 0:
                        iter_num -= 1

                        if prev_point is not None and not current_domain.contains(prev_point):
                            print("!!!!!!!!!!!!!!0")
                            raise ValueError('Wrong prev_point')

                        point = get_random_point(prev_point, polygon,
                                                 parent_structure=parent_structure,
                                                 domain=current_domain)
                        if point is None:
                            iter_num = 0
                            continue

                        is_correct_point = \
                            all([not existing_poly.contains(point) for existing_poly in parent_structure.polygons]) \
                            and not self_intersection(Structure([Polygon('tmp', polygon.points + [point])]))

                    if iter_num == 0:
                        print('Preliminary return of poly')
                        return polygon

                prev_point = point

            polygon.points.append(point)
            # if len(polygon.points) > 2:
            #    polygon.plot()
            #    plt.show()
        # polygon.plot()
        # plt.show()
    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return None
    return polygon


def get_random_point(prev_point: PolygonPoint,
                     parent_poly: Optional[Polygon] = None,
                     parent_structure: Optional[Structure] = None,
                     domain=None) -> Optional[PolygonPoint]:
    if domain is not None:
        current_domain = domain
    else:
        current_domain = GlobalEnv().domain

    if prev_point is not None and not current_domain.contains(prev_point):
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
            pt = PolygonPoint(
                min(max(np.random.normal(prev_point.x, current_domain.len_x * 0.05),
                        current_domain.min_x + current_domain.len_x * 0.05),
                    current_domain.max_x - current_domain.len_x * 0.05),
                min(max(np.random.normal(prev_point.y, current_domain.len_y * 0.05),
                        current_domain.min_y + current_domain.len_y * 0.05),
                    current_domain.max_y - current_domain.len_y * 0.05))
            is_correct_point = current_domain.contains(pt)

            if (is_correct_point and parent_poly and
                    len(parent_poly.points) > 0 and num_iter > MAX_ITER / 2):
                # check then new point is not near existing points
                # if len(parent_poly.points) > 2:
                #    nearest_pts = nearest_points(pt.as_geom(), parent_poly.as_geom())
                #    is_correct_point = nearest_pts[0].distance(nearest_pts[1]) > domains.len_x * 0.05
                # else:
                is_correct_point = all([pt.as_geom().distance(poly_pt.as_geom()) > current_domain.len_x * 0.1
                                        for poly_pt in parent_poly.points])

            if is_correct_point and parent_structure and len(parent_structure.polygons) > 0:
                # check then new point is not near existing polygons
                for poly_from_structure in parent_structure.polygons:
                    if poly_from_structure.length != parent_poly.length:
                        # TODO more smart check
                        _, nearest_pt = nearest_points(pt.as_geom(), poly_from_structure.as_geom().boundary)
                        is_correct_point = pt.as_geom().distance(nearest_pt) > MIN_DIST
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
