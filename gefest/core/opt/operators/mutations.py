import copy
from functools import partial
from typing import Callable
from enum import Enum
import numpy as np
from loguru import logger
from shapely.geometry import LineString, MultiPoint
from shapely.geometry import Point as SPoint

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import get_random_poly, get_selfintersection_safe_point


def mutate_structure(
    structure: Structure,
    domain: Domain,
    operations: list[Callable],
    operation_chance: float,
    operations_probs: list[int],
    **kwargs,
) -> Structure:
    """Apply mutation random mutation from list
        for each polygons in structure.

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
    if len(new_structure.polygons) > (domain.min_poly_num + 1):
        idx_ = idx_ if idx_ else int(np.random.randint(0, len(new_structure)))
        polygon_to_remove = new_structure.polygons[idx_]
        if any([p in polygon_to_remove for p in domain.fixed_points]):
            new_structure.polygons.remove(polygon_to_remove)
    return new_structure


@logger.catch
def add_poly_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
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
    new_structure[idx_] = domain.geometry.resize_poly(
        new_structure[idx_],
        x_scale=np.random.uniform(0.25, 3, 1)[0],
        y_scale=np.random.uniform(0.25, 3, 1)[0],
    )
    return new_structure


@logger.catch
def _get_convex_safe_area(
    poly: Polygon,
    domain: Domain,
    point_left_idx: int,
    point_right_idx: int,
    **kwargs,
) -> Polygon:
    geom = domain.geometry
    if poly[0] == poly[-1]:
        poly = poly[:-1]
    l = len(poly)
    if l == 2:
        p = poly[(point_left_idx + 1) % l]
        circle = SPoint(p.x, p.y).buffer(geom.get_length(poly))
        base_area = [Point(p[0], p[1]) for p in list(circle.exterior.coords)]
        return base_area

    left_cut = [
        poly[(point_left_idx - 1) % l],
        poly[(point_left_idx) % l],
    ]
    right_cut = [
        poly[(point_right_idx + 1) % l],
        poly[(point_right_idx) % l],
    ]
    cut_angles = (
        geom.get_angle(
            left_cut,
            [
                left_cut[0],
                right_cut[0],
            ],
        ),
        geom.get_angle(
            right_cut,
            [
                right_cut[0],
                left_cut[0],
            ],
        ),
    )

    p1, p2 = left_cut[1], right_cut[1]
    pad_vector_points = [p1, geom.rotate_point(p2, p1, 90)]
    pad_vector = (
        pad_vector_points[1].x - pad_vector_points[0].x,
        pad_vector_points[1].y - pad_vector_points[0].y,
    )
    # pad_vector == len(vector[left_point, right_point])
    slice_line = (
        Point(left_cut[1].x + pad_vector[0], left_cut[1].y + pad_vector[1]),
        Point(right_cut[1].x + pad_vector[0], right_cut[1].y + pad_vector[1]),
    )
    scale_factor = max(domain.max_x, domain.max_y) * 100

    if sum(cut_angles) < 170:

        intersection_point = geom.intersection_line_line(
            left_cut,
            right_cut,
            scale_factor,
            scale_factor,
        )
        if intersection_point is not None:
            mid_points = [intersection_point]
        else:
            mid_points = [
                geom.intersection_line_line(left_cut, slice_line, scale_factor, scale_factor),
                geom.intersection_line_line(right_cut, slice_line, scale_factor, scale_factor),
            ]
        try:
            slice_points = geom.intersection_poly_line(
                Polygon(
                    [
                        left_cut[1],
                        *mid_points,
                        right_cut[1],
                    ],
                ),
                slice_line,
                scale_factor,
            )
        except Exception as e:
            raise Exception(e)
            # from shapely.plotting import plot_line
            # from matplotlib import pyplot as plt
            # plot_line(geom._poly_to_shapely_line(poly))
            # plot_line(
            #     LineString(
            #     [(p.x,p.y) for p in
            #     [
            #         left_cut[1],
            #         *mid_points,
            #         right_cut[1],
            #     ]],
            #     ), color='r',
            # )
            # plot_line(LineString([(p.x,p.y) for p in slice_line]), color='g')
            # plt.show()

        if slice_points:
            if isinstance(slice_points, SPoint):
                mid_points = [Point(slice_points.x, slice_points.y)]
            elif isinstance(slice_points, MultiPoint):
                mid_points = [Point(p.x, p.y) for p in slice_points.geoms]
            else:
                mid_points = [Point(p.x, p.y) for p in slice_points.coords]
        base_area = [
            left_cut[1],
            *mid_points,
            right_cut[1],
        ]
        base_area = [
            Point(p[0], p[1])
            for p in geom._poly_to_shapely_poly(Polygon(base_area)).convex_hull.exterior.coords
        ]

    else:
        base_area = [
            left_cut[1],
            geom.intersection_line_line(left_cut, slice_line, scale_factor, scale_factor),
            geom.intersection_line_line(right_cut, slice_line, scale_factor, scale_factor),
            right_cut[1],
        ]

    return Polygon(base_area) if base_area else base_area


@logger.catch
def pos_change_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
) -> Structure:
    geom = domain.geometry
    poly = copy.deepcopy(new_structure[idx_])

    if poly[0] == poly[-1]:
        poly = poly[:-1]
    mutate_point_idx = int(np.random.randint(1, len(poly)))  # fix 1 to 0

    if geom.is_convex:
        poly = geom.get_convex(poly=poly)

    if not geom.is_convex or (len(poly) in (2, 3)):
        point, _ = get_selfintersection_safe_point(
            poly,
            domain,
            mutate_point_idx - 1,
            mutate_point_idx + 1,
        )
        if point:
            poly[mutate_point_idx] = point

    elif geom.is_convex:

        base_area = _get_convex_safe_area(
            poly,
            domain,
            mutate_point_idx - 1,
            mutate_point_idx + 1,
        )

        if base_area:
            movment_area = geom._poly_to_shapely_poly(base_area).intersection(
                geom._poly_to_shapely_poly(domain.allowed_area),
            )
            prohibs = geom.get_prohibited_geom(
                domain.prohibited_area,
                buffer_size=domain.dist_between_polygons,
            )
            for fig in prohibs.geoms:
                movment_area = movment_area.difference(
                    fig.buffer(
                        domain.min_dist_from_boundary,
                    ),
                )

            for idx in [idx for idx in range(len(new_structure)) if idx != idx_]:
                movment_area = movment_area.difference(
                    geom._poly_to_shapely_poly(new_structure[idx]).buffer(
                        domain.dist_between_polygons,
                    ),
                )
            if movment_area.is_empty:
                logger.warning('Empty movment area.')
                return new_structure

            point = geom.get_random_point_in_poly(movment_area)  # pick in geom collection : todo
            if point:
                poly[mutate_point_idx % len(poly)] = point

    if geom.is_closed:
        poly.points.append(poly[0])
    new_structure[idx_] = poly
    if new_structure is None:
        logger.error('None structure.')

    return new_structure


@logger.catch
def add_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
):

    if new_structure is None:
        logger.error('None struct')

    geom = domain.geometry
    poly = copy.deepcopy(new_structure[idx_])
    if geom.is_closed:
        if poly[0] == poly[-1]:
            poly = poly[:-1]
    mutate_point_idx = int(np.random.randint(0, len(poly)))

    if not geom.is_convex or len(poly) == 3:
        point, _ = get_selfintersection_safe_point(
            poly,
            domain,
            mutate_point_idx,
            mutate_point_idx + 1,
        )
        if point:
            poly.points.insert(mutate_point_idx, point)
        else:
            logger.warning('Failed to add point without self intersection.')

    elif geom.is_convex:
        poly = geom.get_convex(poly=poly)
        base_area = _get_convex_safe_area(
            poly,
            domain,
            mutate_point_idx,
            mutate_point_idx + 1,
        )

        if base_area:
            base_area = geom._poly_to_shapely_poly(Polygon(base_area))
            if not base_area.is_simple:
                logger.error('Base area not simple.')

            movment_area = base_area.intersection(
                geom._poly_to_shapely_poly(domain.allowed_area),
            )

            prohibs = geom.get_prohibited_geom(
                domain.prohibited_area,
                buffer_size=domain.dist_between_polygons,
            )
            for fig in prohibs.geoms:
                movment_area = movment_area.difference(
                    fig.buffer(
                        domain.min_dist_from_boundary,
                    ),
                )

            for idx in [idx for idx in range(len(new_structure)) if idx != idx_]:
                movment_area = movment_area.difference(
                    geom._poly_to_shapely_poly(new_structure[idx]).buffer(
                        domain.dist_between_polygons,
                    ),
                )
            if movment_area.is_empty:
                logger.warning('Empty movment area')
                return new_structure
            else:
                pass
                # logger.warning('Not implemented select adjacent to poly movment_area part.')
                # logger.warning('Not implemented number of parts check. If there is 1 part - ok.')
            point = geom.get_random_point_in_poly(movment_area)
            if point:
                if mutate_point_idx + 1 < len(poly):
                    poly.points.insert(
                        mutate_point_idx + 1,
                        point,
                    )
                else:
                    poly.points.insert(
                        mutate_point_idx - 1,
                        point,
                    )

    if geom.is_closed:
        poly.points.append(poly[0])
    new_structure[idx_] = poly
    if new_structure is None:
        logger.error('None struct')
    return new_structure


@logger.catch
def drop_point_mutation(
    new_structure: Structure,
    domain: Domain,
    idx_: int = None,
    **kwargs,
):

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
    rotate_poly = partial(rotate_poly_mutation)
    resize_poly = partial(resize_poly_mutation)
    add_point = partial(add_point_mutation)
    drop_point = partial(drop_point_mutation)
    add_poly = partial(add_poly_mutation)
    drop_poly = partial(drop_poly_mutation)
    pos_change_point = partial(pos_change_point_mutation)
