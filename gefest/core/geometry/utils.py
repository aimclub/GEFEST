from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gefest.core.geometry.domain import Domain

from random import randint
from typing import Optional

import numpy as np
from polygenerator import (
    random_convex_polygon,
    random_polygon,
    random_star_shaped_polygon,
)
from shapely.affinity import scale
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
)
from shapely.geometry import Point as SPoint

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry2D


def random_polar(origin: Point, radius_scale: float) -> Point:
    """Generates random point in circe.

    The distribution density is shifted to the center.
    https://habrastorage.org/r/w1560/webt/sn/xx/ow/snxxowuhnuqnr8dp4sadmyswqu0.png
    """
    theta = np.random.random() * 2 * np.pi
    r = np.random.random() * radius_scale
    return Point((r * np.cos(theta)) + origin.x, (r * np.sin(theta)) + origin.y)


def get_random_structure(domain: Domain, **kwargs) -> Structure:
    """Generates random structure."""
    structure = Structure(polygons=())
    num_pols = randint(domain.min_poly_num, domain.max_poly_num)

    for _ in range(num_pols):
        polygon = get_random_poly(parent_structure=structure, domain=domain)
        if polygon is not None and len(polygon.points) > 1:
            structure.append(polygon)
        else:
            continue

    return structure


def get_random_poly(parent_structure: Optional[Structure], domain: Domain) -> Optional[Polygon]:
    """Generates random polygon."""
    geometry = domain.geometry
    try:
        """
        Function for creation random polygon.
        The main idea is to create centroids along with a neighborhood to locate the polygon.
        Neighborhood sizes range from small to large.
        The main condition for creating a neighborhood is the absence of other polygons in it.
        After setting the neighborhood, polygons are created around the centroid inside it.
        This approach is less demanding on postprocessing than random creation
        """
        occupied_area = _create_area(domain, parent_structure, geometry)
        if occupied_area is None:
            # If it was not possible to find the occupied area then returns None
            return None
        else:
            centroid = occupied_area[0]
            sigma = occupied_area[1]  # Size of neighborhood
            # The polygon is created relative to the centroid
            # and the size of the neighborhood
            polygon = create_poly(centroid, sigma, domain, geometry)
    except Exception as ex:
        print(ex)
        import traceback

        print(traceback.format_exc())
        return None

    return polygon


def create_poly(centroid: Point, sigma: int, domain: Domain, geometry: Geometry2D) -> Polygon:
    """Generates random polygon using poltgenerator lib.

    For details see: https://github.com/bast/polygenerator
    """
    num_points = randint(
        domain.min_points_num,
        domain.max_points_num,
    )
    if domain.geometry.is_convex:  # convex closed/unclosed
        generator = random_convex_polygon

    else:  # non_convex, unclosed
        generator = np.random.choice(
            [
                random_convex_polygon,
                random_star_shaped_polygon,
                random_polygon,
            ],
        )

    new_poly = generator(num_points)
    if not geometry.is_closed:
        start = np.random.choice(range(num_points), 1)[0]
        new_poly = [new_poly[((start + i) % len(new_poly))] for i in range(num_points)]

    scale_factor = 2 * (sigma / (1 ** 0.5))
    c, s = centroid, scale_factor
    new_poly = Polygon(
        [
            Point(
                ((p[0] - 0.5) * s) + c.x,
                ((p[1] - 0.5) * s) + c.y,
            )
            for p in new_poly
        ],
    )
    if domain.geometry.is_closed:
        new_poly.points.append(new_poly[0])

    return new_poly


def _get_sigma_max(poly, init_max):
    sigma_max = init_max
    left_bound = 0
    right_bound = init_max
    for _ in range(100):
        if poly.buffer(-sigma_max, 1).is_empty:
            right_bound = sigma_max
            sigma_max -= (right_bound - left_bound) / 2
        else:
            left_bound = sigma_max
            sigma_max += (right_bound - left_bound) / 2

        if sigma_max < 0.02:
            break

    return sigma_max


def _create_area(domain: Domain, structure: Structure, geometry: Geometry2D) -> (Point, float):
    """Finds free area for new polygon."""
    geom = domain.geometry
    area = geom._poly_to_shapely_poly(domain.bound_poly).buffer(-(domain.min_dist_from_boundary), 1)
    prohibs = geom.get_prohibited_geom(
        domain.prohibited_area,
        buffer_size=domain.dist_between_polygons,
    )
    for g in prohibs.geoms:
        area = area.difference(
            g.buffer(
                domain.min_dist_from_boundary,
            ),
        )

    for poly in structure.polygons:
        area = area.difference(
            geom._poly_to_shapely_poly(poly).convex_hull.buffer(domain.dist_between_polygons, 1),
        ).intersection(area)

    sigma_max = 0.95 * _get_sigma_max(area, (min(domain.max_x, domain.max_y) / 2) * 1.01)
    sigma_min = max(domain.max_x - domain.min_x, domain.max_y - domain.min_y) * 0.05

    sigma = np.random.uniform(sigma_min, sigma_max)
    centroid = geom.get_random_point_in_shapey_geom(area.buffer(-sigma, 1))

    return centroid, sigma * 0.99


def get_selfintersection_safe_point(
    poly: Polygon,
    domain: Domain,
    point_left_idx: int,
    point_right_idx: int,
) -> Polygon:
    """Finds a new point for the polygon that does not generate self-intersections."""
    geom = domain.geometry

    if geom.is_closed:
        parts = [
            *poly.points[(point_right_idx) % len(poly) : :],
            *poly.points[0 : (point_left_idx + 1) % len(poly)],
        ]
        border = [geom._poly_to_shapely_line(Polygon(parts))]
    else:
        border = [
            geom._poly_to_shapely_line(Polygon(part))
            for part in [
                poly.points[(point_right_idx) % len(poly) :],
                poly.points[0 : (point_left_idx + 1) % len(poly)],
            ]
            if len(part) != 1
        ]

    border = MultiLineString(
        [scale(line, xfact=0.99, yfact=0.99) for line in border if not line.is_empty],
    )
    l_ = poly[point_left_idx % len(poly)]
    r_ = poly[point_right_idx % len(poly)]

    origin = Point((l_.x + r_.x) / 2, (l_.y + l_.y) / 2)
    scalefactor = (
        (LineString(((l_.x, l_.y), (r_.x, r_.y))).length / 2)
        if len(poly) > 2
        else geom.get_length(poly) * 1.5
    )
    points = []
    p_area = geom.get_prohibited_geom(
        domain.prohibited_area,
        buffer_size=domain.dist_between_polygons,
    )
    for _ in range(200):
        point = random_polar(origin, scalefactor)
        points.append(point)
        new_segment = scale(
            LineString(((l_.x, l_.y), (point.x, point.y), (r_.x, r_.y))),
            0.99,
            0.99,
        )

        if all(
            (
                not new_segment.intersects(border),
                not new_segment.intersects(geom._poly_to_shapely_line(domain.allowed_area)),
                not any(g.intersects(new_segment) for g in p_area.geoms),
            ),
        ):
            break
    else:
        point = None

    return point, border


def get_convex_safe_area(
    poly: Polygon,
    domain: Domain,
    point_left_idx: int,
    point_right_idx: int,
    structure: Structure,
    poly_idx: int,
    **kwargs,
) -> Polygon:
    """Finds an area from which a new point can be selected without breaking the convexity.

    Point_left_idx and point_right_idx expected to be neighbours.
    """
    geom = domain.geometry
    movment_area = None

    poly_len = len(poly)
    if poly_len == 2:
        p = poly[(point_left_idx + 1) % poly_len]
        circle = SPoint(p.x, p.y).buffer(geom.get_length(poly) * 1.5)
        base_area = [Point(p[0], p[1]) for p in list(circle.exterior.coords)]

    elif poly_len == 3:
        p = poly[(point_left_idx + 1) % poly_len]
        circle = SPoint(p.x, p.y).buffer(geom.get_length(poly) / 3)
        base_area = [Point(p[0], p[1]) for p in list(circle.exterior.coords)]

    else:
        left_cut = [
            poly[((point_left_idx - 1) + poly_len) % poly_len],
            poly[(point_left_idx) % poly_len],
        ]
        right_cut = [
            poly[(point_right_idx + 1) % poly_len],
            poly[(point_right_idx) % poly_len],
        ]

        cut_angles = (
            geom.get_angle(
                left_cut,
                (left_cut[1], right_cut[1]),
            ),
            geom.get_angle(
                right_cut,
                (right_cut[1], left_cut[1]),
            ),
        )

        p1, p2 = left_cut[1], right_cut[1]
        pad_vector_points = [p1, geom.rotate_point(p2, p1, -90)]

        pad_vector = (
            pad_vector_points[1].x - pad_vector_points[0].x,
            pad_vector_points[1].y - pad_vector_points[0].y,
        )

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
                base_area = [
                    left_cut[1],
                    intersection_point,
                    right_cut[1],
                ]
            else:
                base_area = [
                    left_cut[1],
                    geom.intersection_line_line(left_cut, slice_line, scale_factor, scale_factor),
                    geom.intersection_line_line(right_cut, slice_line, scale_factor, scale_factor),
                    right_cut[1],
                ]

            slice_points = geom.intersection_poly_line(
                Polygon(base_area),
                slice_line,
                scale_factor,
            )
            if slice_points:
                base_area = [
                    left_cut[1],
                    *slice_points,
                    right_cut[1],
                ]
                base_area = [
                    Point(p[0], p[1])
                    for p in geom._poly_to_shapely_poly(
                        Polygon(base_area),
                    ).convex_hull.exterior.coords
                ]

        else:
            base_area = [
                left_cut[1],
                geom.intersection_line_line(left_cut, slice_line, scale_factor, scale_factor),
                geom.intersection_line_line(right_cut, slice_line, scale_factor, scale_factor),
                right_cut[1],
            ]

            if None in base_area:
                return None

            if not geom._poly_to_shapely_poly(Polygon(base_area)).is_simple:
                base_area = [
                    left_cut[1],
                    geom.intersection_line_line(left_cut, right_cut, scale_factor, scale_factor),
                    right_cut[1],
                ]

        if base_area:
            other_polygons = Structure([poly for i, poly in enumerate(structure) if i != poly_idx])
            movment_area = _substract_oссupied_area(
                Polygon(base_area),
                other_polygons,
                domain,
                left_cut[1].coords,
                right_cut[1].coords,
            )

    return movment_area


def _substract_oссupied_area(
    base_area: Polygon,
    structure: Structure,
    domain: Domain,
    left_point: tuple[float, float],
    right_point: tuple[float, float],
):
    geom = domain.geometry
    prohibs = geom.get_prohibited_geom(
        domain.prohibited_area,
        buffer_size=domain.dist_between_polygons,
    )

    movment_area = geom._poly_to_shapely_poly(base_area).intersection(
        geom._poly_to_shapely_poly(domain.allowed_area),
    )

    for fig in prohibs.geoms:
        movment_area = movment_area.difference(
            fig.buffer(
                domain.min_dist_from_boundary,
            ),
        )

    for idx in range(len(structure)):
        movment_area = movment_area.difference(
            geom._poly_to_shapely_poly(structure[idx]).buffer(
                domain.dist_between_polygons,
            ),
        )

    if movment_area.is_empty:
        movment_area = None
    elif isinstance(movment_area, (MultiPolygon, GeometryCollection)):
        for g_ in movment_area.geoms:
            if g_.intersects(LineString((left_point, right_point))):
                movment_area = g_
                break

    return movment_area
