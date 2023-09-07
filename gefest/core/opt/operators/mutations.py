import copy
import random
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.geometry import Point, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import get_random_point, get_random_poly


def mutate_structure(
    structure: Structure,
    domain: Domain,
    mutations: list[Callable],
    mutation_chance: float,
    mutations_probs: list[int],
):

    new_structure = copy.deepcopy(structure)

    for _ in enumerate(range(len(new_structure))):
        idx_ = np.random.randint(0, len(new_structure))
        if np.random.random() < mutation_chance:
            chosen_mutation = np.random.choice(
                a=mutations,
                size=1,
                p=mutations_probs,
            )
            new_structure = chosen_mutation[0](new_structure, domain, idx_)


def rotate_poly(new_structure: Structure, domain: Domain, idx_: int = None) -> Structure:
    angle = float(np.random.randint(-120, 120))
    new_structure.polygons[idx_] = domain.geometry.rotate_poly(
        new_structure.polygons[idx_],
        angle,
    )
    return new_structure


def drop_poly(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
) -> Structure:
    if len(new_structure.polygons) > (domain.min_poly_num + 1):
        idx_ = idx_ if idx_ else int(np.random.randint(0, len(new_structure)))
        polygon_to_remove = new_structure.polygons[idx_]
        if any([p in polygon_to_remove for p in domain.fixed_points]):
            new_structure.polygons.remove(polygon_to_remove)
    return new_structure


def add_poly(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
) -> Structure:
    if len(new_structure.polygons) < (domain.max_poly_num - 1):
        new_poly = get_random_poly(new_structure, domain)
        if new_poly is not None:
            new_structure.polygons.append(new_poly)
    return new_structure


def resize_poly(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
) -> Structure:
    new_structure.polygons[idx_] = domain.geometry.resize_poly(
        new_structure.polygons[idx_],
        x_scale=np.random.uniform(0.25, 3, 1)[0],
        y_scale=np.random.uniform(0.25, 3, 1)[0],
    )
    return new_structure


def pos_change_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
) -> Structure:

    mutate_point_idx = int(np.random.randint(0, len(new_structure[idx_])))
    # Neighborhood to reposition
    eps_x = round(domain.len_x / 10)
    eps_y = round(domain.len_y / 10)

    structure = copy.deepcopy(new_structure)

    # Displacement in the neighborhood
    displacement_x = random.randint(-eps_x, eps_x)
    displacement_y = random.randint(-eps_y, eps_y)

    x_new = structure.polygons[idx_].points[mutate_point_idx].x + displacement_x
    y_new = structure.polygons[idx_].points[mutate_point_idx].y + displacement_y

    i = 20  # Number of attempts to change the position of the point
    while Point(x_new, y_new) not in domain:
        x_new = structure.polygons[idx_].points[mutate_point_idx].x + displacement_x
        y_new = structure.polygons[idx_].points[mutate_point_idx].y + displacement_y
        i -= 1
        if i == 0:
            return new_structure

    structure.polygons[idx_].points[mutate_point_idx].x = x_new
    structure.polygons[idx_].points[mutate_point_idx].y = y_new

    # from gefest.core.viz.struct_vizualizer import StructVizualizer
    # from matplotlib import pyplot as plt

    # plt.figure(figsize=(7, 7))
    # visualiser = StructVizualizer(domain)

    # info = {
    #     'spend_time': 1,
    #     'fitness': 0,
    #     'type': 'prediction',
    # }
    # visualiser.plot_structure(
    #     [structure, new_structure], [info, info], ['-', '-.'],
    # )

    # plt.show(block=True)

    return structure


def add_point(new_structure: Structure, domain: Domain, idx_: int = None):
    mutate_point_idx = int(np.random.randint(0, len(new_structure[idx_])))

    polygon_to_mutate = new_structure[idx_]

    new_point = get_random_point(
        polygon_to_mutate,
        new_structure,
        domain,
    )

    if new_point is not None:
        if mutate_point_idx + 1 < len(polygon_to_mutate):
            new_structure.polygons[idx_].points.insert(
                mutate_point_idx + 1,
                new_point,
            )
        else:
            new_structure.polygons[idx_].points.insert(
                mutate_point_idx - 1,
                new_point,
            )
    return new_structure


def drop_point(new_structure: Structure, domain: Domain, idx_: int = None):
    mutate_point_idx = int(np.random.randint(0, len(new_structure[idx_])))

    polygon_to_mutate = new_structure[idx_]
    point_to_mutate = polygon_to_mutate[mutate_point_idx]

    if len(polygon_to_mutate) > domain.min_points_num:
        # if drop point from polygon
        new_structure.polygons[idx_].points.remove(point_to_mutate)

    return new_structure


# class BaseMutations(Enum):
#     mutation_name = mutation
