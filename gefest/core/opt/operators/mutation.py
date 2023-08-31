from copy import deepcopy
from multiprocessing import Pool

import numpy as np

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.geometry import Polygon, Structure
from gefest.core.geometry.utils import get_random_point, get_random_poly
from gefest.core.opt.domain import Domain
from gefest.core.structure.point import Point


def rotate_poly(new_structure: Structure, domain: Domain) -> Structure:
    angle = float(np.random.randint(-120, 120))
    new_structure.polygons[0] = domain.geometry.rotate_poly(
        new_structure.polygons[0],
        angle,
    )
    return new_structure


def drop_poly(
    new_structure: Structure,
    domain: Domain,
) -> Structure:
    if len(new_structure.polygons) > (domain.min_poly_num + 1):
        polygon_to_mutate_idx = int(np.random.randint(0, len(new_structure)))
        polygon_to_remove = new_structure.polygons[polygon_to_mutate_idx]
        if any([p in polygon_to_remove for p in domain.fixed_points]):
            new_structure.polygons.remove(polygon_to_remove)
    return new_structure


def add_poly(
    new_structure: Structure,
    domain: Domain,
) -> Structure:
    if len(new_structure.polygons) < (domain.max_poly_num - 1):
        new_poly = get_random_poly(new_structure, domain)
        if new_poly is not None:
            new_structure.polygons.append(new_poly)
    return new_structure


def resize_poly(new_structure: Structure, domain: Domain) -> Structure:
    polygon_to_mutate_idx = int(np.random.randint(0, len(new_structure.polygons)))
    new_structure.polygons[polygon_to_mutate_idx] = domain.geometry.resize_poly(
        new_structure.polygons[polygon_to_mutate_idx],
        x_scale=np.random.uniform(0.25, 3, 1)[0],
        y_scale=np.random.uniform(0.25, 3, 1)[0],
    )
    return new_structure


def add_point(new_structure: Structure, domain: Domain):
    val = len(new_structure.polygons)
    polygon_to_mutate_idx = int(np.random.randint(0, len(new_structure.polygons)))
    val = len(new_structure[polygon_to_mutate_idx])
    mutate_point_idx = int(np.random.randint(0, len(new_structure[polygon_to_mutate_idx])))

    polygon_to_mutate = new_structure[polygon_to_mutate_idx]
    point_to_mutate = polygon_to_mutate[mutate_point_idx]

    new_point = get_random_point(
        polygon_to_mutate,
        new_structure,
        domain,
    )

    if new_point is not None:
        if mutate_point_idx + 1 < len(polygon_to_mutate):
            new_structure.polygons[polygon_to_mutate_idx].points.insert(
                mutate_point_idx + 1, new_point
            )
        else:
            new_structure.polygons[polygon_to_mutate_idx].points.insert(
                mutate_point_idx - 1, new_point
            )
    return new_structure


def drop_point(new_structure: Structure, domain: Domain):
    polygon_to_mutate_idx = int(np.random.randint(0, len(new_structure.polygons)))
    mutate_point_idx = int(np.random.randint(0, len(new_structure[polygon_to_mutate_idx])))

    polygon_to_mutate = new_structure[polygon_to_mutate_idx]
    point_to_mutate = polygon_to_mutate[mutate_point_idx]

    if len(polygon_to_mutate) > domain.min_points_num:
        # if drop point from polygon
        new_structure.polygons[polygon_to_mutate_idx].points.remove(point_to_mutate)

    return new_structure
