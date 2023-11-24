import copy
import random
from enum import Enum
from functools import partial
from itertools import product
from typing import Callable

import numpy as np
from loguru import logger

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.utils import where


def crossover_structures(
    structure1: Structure,
    structure2: Structure,
    domain: Domain,
    operations: list[Callable],
    operation_chance: float,
    operations_probs: list[int],
    **kwargs,
) -> tuple[Structure]:
    """Applys random crossover from given list for pair of structures.

    Args:
        structure1 (Structure): First parent.
        structure1 (Structure): Second parent.
        domain (Domain): Task domain.
        operations (list[Callable]): List of crossovers operations to choose.
        operation_chance (float): Chance of crossover.
        operations_probs (list[int]): Probablilites of each crossover operation.

    Returns:
        tuple[Structure]: Сhildren.
    """
    s1, s2 = copy.deepcopy(structure1), copy.deepcopy(structure2)
    chosen_crossover = np.random.choice(
        a=operations,
        size=1,
        p=operations_probs,
    )
    new_structure = chosen_crossover[0](s1, s2, domain)
    if not new_structure:
        logger.warning(f'None out: {chosen_crossover[0].__name__}')

    return new_structure


# pairs for crossover selection
def panmixis(pop: list[Structure]) -> list[tuple[Structure, Structure]]:
    """Default pair selection strategy."""
    np.random.shuffle(list(pop))
    return [(pop[idx], pop[idx + 1]) for idx in range(len(pop) - 1)]


# best indivisual selection
def structure_level_crossover(
    s1: Structure,
    s2: Structure,
    domain: Domain,
    **kwargs,
):
    """Exchanges points of two polygons."""
    s1, s2 = copy.deepcopy(s1), copy.deepcopy(s2)
    polygons1 = s1.polygons
    polygons2 = s2.polygons
    crossover_point = np.random.randint(
        0,
        len(polygons1) + 1,
    )

    # Crossover conversion
    part_1 = polygons1[0:crossover_point]
    if not isinstance(part_1, tuple):
        part_1 = part_1

    part_2 = polygons2[crossover_point : len(s1.polygons)]
    if not isinstance(part_2, tuple):
        part_2 = part_2

    result = list(copy.deepcopy(part_1))
    result.extend(copy.deepcopy(part_2))

    new_structure = Structure(polygons=result)

    return (new_structure,)


def polygon_level_crossover(
    s1: Structure,
    s2: Structure,
    domain: Domain,
    **kwargs,
):
    """Exchanges points of two nearest polygons in structure."""
    geom = domain.geometry
    s1, s2 = copy.deepcopy(s1), copy.deepcopy(s2)
    intersected = False
    split_angle = 0
    pairs_dists = [
        (p, geom.min_distance(p[0], p[1]))
        for p in list(product(s1, s2))
        if (p[0] is not p[1] and len(p[0]) != 0 and len(p[1]) != 0)
    ]
    intersects_ids = where(pairs_dists, lambda p_d: p_d[1] == 0)
    if len(intersects_ids) > 0:
        intersected = True
        pairs_dists = [p_d for idx_, p_d in enumerate(pairs_dists) if idx_ in intersects_ids]
        pairs_dists = [
            (p[0], geom.min_distance(geom.get_centroid(p[0][0]), geom.get_centroid(p[0][1])))
            for p in pairs_dists
        ]

    pairs_dists = sorted(pairs_dists, key=lambda p_d: p_d[1])
    if len(pairs_dists) == 0:
        return (s1,)

    poly_1 = pairs_dists[0][0][0]
    poly_2 = pairs_dists[0][0][1]
    if intersected:
        # now not adaptive angle #
        split_angle = (np.random.rand() * 2 - 1) * (70)
    elif pairs_dists[0][1] > domain.dist_between_polygons:
        return (s1,)

    c1, c2 = geom.get_centroid(poly_1), geom.get_centroid(poly_2)
    vector1 = geom.rotate_point(point=c2, origin=c1, angle=split_angle)
    vector2 = geom.rotate_point(point=c1, origin=c2, angle=split_angle)
    scale_factor = max(domain.max_x, domain.max_y)
    parts_1 = geom.split_polygon(poly_1, [c1, vector1], scale_factor)
    parts_2 = geom.split_polygon(poly_2, [c2, vector2], scale_factor)
    if len(parts_1) < 2 or len(parts_2) < 2:
        return (s1,)

    new_parts = (*parts_1[0], *parts_2[1]) if c1.y > c2.y else (*parts_1[1], *parts_2[0])
    new_poly = geom.get_convex(Polygon(points=[Point(*p) for p in list(set(new_parts))]))
    if len(new_poly) > domain.max_points_num:
        random_elements = random.choice(new_poly.points)
        new_poly.points = list(filter(lambda x: x != random_elements, new_poly.points))

    idx_ = where(s1.polygons, lambda p: p == poly_1)[0]
    s1[idx_] = new_poly
    return (s1,)


class CrossoverTypes(Enum):
    """Enumerates all crossover functions."""

    structure_level = partial(structure_level_crossover)
    polygon_level = partial(polygon_level_crossover)
