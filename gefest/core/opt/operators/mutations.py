import copy
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np
from loguru import logger
from shapely.geometry import LineString

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import (
    get_convex_safe_area,
    get_random_poly,
    get_selfintersection_safe_point,
)


def mutate_structure(
    structure: Structure,
    domain: Domain,
    operations: list[Callable],
    operation_chance: float,
    operations_probs: list[float],
    **kwargs,
) -> Structure:
    """Applys random mutation from given list for each polygon in structure.

    Args:
        structure (Structure): Structure to mutate.
        domain (Domain): Task domain.
        mutations (list[Callable]): List of mutation operations to choose.
        mutation_chance (float): Chance to mutate polygon.
        mutations_probs (list[int]): Probablilites of each mutation operation.

    Returns:
        Structure: Mutated structure. It is not guaranteed
            that the resulting structure will be valid or changed.
    """
    new_structure = copy.deepcopy(structure)

    for _ in enumerate(range(len(new_structure))):
        idx_ = np.random.randint(0, len(new_structure))
        if np.random.random() < operation_chance:
            chosen_mutation = np.random.choice(
                a=operations,
                size=1,
                p=operations_probs,
            )
            new_structure = chosen_mutation[0](new_structure, domain, idx_)
            if not new_structure:
                logger.warning(f'None out: {chosen_mutation[0].__name__}')

    return new_structure


@logger.catch
def rotate_poly_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
    """Rotares polygon for random angle."""
    angle = float(np.random.randint(-120, 120))
    new_structure[idx_] = domain.geometry.rotate_poly(
        new_structure[idx_],
        angle,
    )
    return new_structure


@logger.catch
def drop_poly_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
    """Drops random polygon from structure."""
    if len(new_structure.polygons) > (domain.min_poly_num + 1):
        idx_ = idx_ if idx_ else int(np.random.randint(0, len(new_structure)))
        polygon_to_remove = new_structure.polygons[idx_]
        if not any(p in polygon_to_remove for p in domain.fixed_points):
            new_structure.remove(polygon_to_remove)

    return new_structure


@logger.catch
def add_poly_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
    """Adds random polygon into structure using standard generator."""
    if len(new_structure) < (domain.max_poly_num - 1):
        new_poly = get_random_poly(new_structure, domain)
        if new_poly is not None:
            new_structure.append(new_poly)

    return new_structure


@logger.catch
def resize_poly_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
    """Randomly resizes polygon."""
    new_structure[idx_] = domain.geometry.resize_poly(
        new_structure[idx_],
        x_scale=np.random.uniform(0.25, 3, 1)[0],
        y_scale=np.random.uniform(0.25, 3, 1)[0],
    )
    return new_structure


@logger.catch
def pos_change_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
    """Moves a random point without violating the geometry type specified in the domain."""
    geom = domain.geometry
    poly = copy.deepcopy(new_structure[idx_])
    if poly[0] == poly[-1]:
        poly = poly[:-1]

    mutate_point_idx = int(np.random.randint(0, len(poly)))
    new_point = None
    if not geom.is_convex or (len(poly) in (2, 3)):
        new_point, _ = get_selfintersection_safe_point(
            poly,
            domain,
            mutate_point_idx - 1,
            mutate_point_idx + 1,
        )

    elif geom.is_convex:
        poly = geom.get_convex(poly=poly)[:-1]
        movment_area = get_convex_safe_area(
            poly,
            domain,
            mutate_point_idx - 1,
            mutate_point_idx + 1,
            new_structure,
            idx_,
        )

        if movment_area:
            if not movment_area:
                return new_structure
            else:
                new_point = geom.get_random_point_in_poly(movment_area)

    else:
        logger.warning('Strange case')

    if new_point:
        poly[mutate_point_idx % len(poly)] = new_point

    if geom.is_closed:
        poly.points.append(poly[0])

    new_structure[idx_] = poly
    return new_structure


@logger.catch
def add_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
):
    """Adds a random point without violating the geometry type specified in the domain."""
    geom = domain.geometry
    poly = copy.deepcopy(new_structure[idx_])

    if poly[0] == poly[-1]:
        poly = poly[:-1]

    mutate_point_idx = int(np.random.randint(0, len(poly)))
    new_point = None
    if not geom.is_convex or len(poly) == 3:
        new_point, _ = get_selfintersection_safe_point(
            poly,
            domain,
            mutate_point_idx,
            mutate_point_idx + 1,
        )

    elif geom.is_convex:
        poly = geom.get_convex(poly=poly)[:-1]
        movment_area = get_convex_safe_area(
            poly,
            domain,
            mutate_point_idx,
            mutate_point_idx + 1,
            new_structure,
            idx_,
        )

        if movment_area:
            if not movment_area:
                return new_structure
            else:
                new_point = geom.get_random_point_in_poly(movment_area)

    else:
        logger.warning('Strange case')

    if new_point:
        poly.points.insert(
            (mutate_point_idx + 1) % len(poly),
            new_point,
        )

    if geom.is_closed:
        poly.points.append(poly[0])

    new_structure[idx_] = poly
    return new_structure


@logger.catch
def drop_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
):
    """Drops random point from polygon."""
    polygon_to_mutate = new_structure[idx_]
    if domain.geometry.is_closed:
        if polygon_to_mutate[0] == polygon_to_mutate[-1]:
            polygon_to_mutate = polygon_to_mutate[:-1]

    mutate_point_idx = int(np.random.randint(0, len(polygon_to_mutate)))
    point_to_mutate = polygon_to_mutate[mutate_point_idx]

    if len(polygon_to_mutate) > domain.min_points_num:
        if domain.geometry.is_closed or idx_ == 0 or idx_ == (len(polygon_to_mutate) - 1):
            polygon_to_mutate.points.remove(point_to_mutate)
        else:
            new_poly = [
                polygon_to_mutate[idx]
                for idx in range(len(polygon_to_mutate))
                if idx != mutate_point_idx
            ]
            if LineString([(p.x, p.y) for p in new_poly]).is_simple:
                polygon_to_mutate.points.remove(point_to_mutate)

    new_structure[idx_] = polygon_to_mutate
    return new_structure


class MutationTypes(Enum):
    """enumerates all mutation functions."""

    rotate_poly = partial(rotate_poly_mutation)
    resize_poly = partial(resize_poly_mutation)
    add_point = partial(add_point_mutation)
    drop_point = partial(drop_point_mutation)
    add_poly = partial(add_poly_mutation)
    drop_poly = partial(drop_poly_mutation)
    pos_change_point = partial(pos_change_point_mutation)
