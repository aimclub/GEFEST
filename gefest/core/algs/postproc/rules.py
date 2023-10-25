import copy
from enum import Enum
from itertools import combinations

import numpy as np
from loguru import logger
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity

from gefest.core.algs.postproc.rules_base import PolygonRule, StructureRule
from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain


class PolygonsNotTooClose(StructureRule):
    @staticmethod
    def validate(struct: Structure, domain: Domain) -> bool:
        pairs = tuple(combinations(struct.polygons, 2))
        is_too_close = [False] * len(pairs)

        for idx, pair in enumerate(pairs):
            is_too_close[idx] = (
                _pairwise_dist(pair[0], pair[1], domain) < domain.dist_between_polygons
            )

        return not any(is_too_close)

    @staticmethod
    @logger.catch
    def correct(struct: Structure, domain: Domain) -> Structure:
        """
        For polygons that are closer than the specified threshold,
        one of them will be removed
        """
        polygons = struct.polygons
        num_poly = len(polygons)
        to_delete = []

        for i in range(num_poly - 1):
            for j in range(i + 1, num_poly):
                distance = _pairwise_dist(polygons[i], polygons[j], domain)
                if distance < domain.dist_between_polygons:
                    if (
                        polygons[i] not in domain.fixed_points
                        or polygons[i] not in domain.prohibited_area
                    ):
                        to_delete.append(i)  # Collecting polygon indices for deletion

        to_delete_poly = [struct.polygons[i] for i in np.unique(to_delete)]
        corrected_structure = Structure(
            polygons=[poly for poly in struct.polygons if poly not in to_delete_poly],
        )

        return corrected_structure


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    """
    ::TODO:: find the answer for the question: why return 0 gives infinite computation
    """
    if poly_1 is poly_2 or len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    # nearest_pts = domain.geometry.nearest_points(poly_1, poly_2) ??? why return only 1 point
    return domain.geometry.min_distance(poly_1, poly_2)


class PointsNotTooClose(PolygonRule):
    @staticmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        """The method indicates that any :obj:`Point` in each :obj:`Polygon` of :obj:`Structure`
        is placed in correct distance by previous point
        Args:
            structure: the :obj:`Structure` that explore
        Returns:
            ``True`` if any side of poly have incorrect lenght, otherwise - ``False``
        """
        poly = structure[idx_poly_with_error]
        if poly[0] != poly[-1] and domain.geometry.is_closed:
            poly[:-1] = poly[0]
        lenght = domain.dist_between_points
        check, norms = [[None] * (len(poly) - 1)] * 2
        for idx, pair in enumerate(
            zip(
                poly[:-1],
                poly[1:],
            ),
        ):
            norm = np.linalg.norm(np.array(pair[1].coords) - np.array(pair[0].coords))
            norms[idx] = norm
            check[idx] = norm > lenght
        return all(check)

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        poly = copy.deepcopy(structure[idx_poly_with_error])
        poly = domain.geometry.simplify(poly, domain.dist_between_points * 1.05)

        if poly[0] != poly[-1] and domain.geometry.is_closed:
            poly.points.append(poly[0])
        elif poly[0] == poly[-1] and not domain.geometry.is_closed:
            poly.points = poly.points[:-1]
        return poly


class PolygonNotOverlapsProhibited(PolygonRule):
    @staticmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:

        geom = domain.geometry
        if domain.geometry.is_closed:
            # logger.warning("There is no errors if no construction. NotImplemented validation and fix.")
            pass
        else:
            from matplotlib import pyplot as plt
            from shapely.plotting import plot_line, plot_polygon

            prohib = geom.get_prohibited_geom(domain.prohibited_area, domain.dist_between_polygons)
            prohib = unary_union(prohib)
            poly = geom._poly_to_shapely_line(structure[idx_poly_with_error])
            # for g in prohib.geoms:
            #     plot_polygon(g)
            # plot_line(poly, color='r')
            # plt.show(block=True)
            if poly.intersects(prohib):
                return False

        return True

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        geom = domain.geometry
        if domain.geometry.is_closed:
            raise NotImplementedError()
        else:
            from matplotlib import pyplot as plt
            from shapely.plotting import plot_line, plot_polygon

            prohib = geom.get_prohibited_geom(domain.prohibited_area, domain.dist_between_polygons)
            # for g in prohib.geoms:
            # plot_polygon(g)
            # plt.show(block=True)
            prohib = unary_union(prohib)
            # for g in prohib.geoms:
            #     plot_polygon(g)

            poly = geom._poly_to_shapely_line(structure[idx_poly_with_error])

            # plot_line(poly)
            # plt.show(block=True)
            if poly.intersects(prohib):
                res = poly.difference(prohib.buffer(0.001))
                from shapely.geometry import GeometryCollection, LineString, MultiPoint

                if isinstance(res, (MultiPoint, LineString)):
                    res = GeometryCollection(res)
                parts = [g for g in res.geoms]
                parts = [g for g in parts if not g.intersects(prohib)]
                poly = np.random.choice(parts)
                # plot_line(poly)
                # poly = [Point(p[0], p[1]) for p in poly.coords]
                # plt.show(block=True)
                return Polygon([Point(p[0], p[1]) for p in poly.coords])
            else:
                return Polygon([Point(p[0], p[1]) for p in poly.coords])


class PolygonGeometryIsValid(PolygonRule):
    @staticmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        poly = structure[idx_poly_with_error]
        if (domain.geometry.is_closed and (poly[0] == poly[-1])) or (
            not domain.geometry.is_closed and (poly[0] != poly[-1])
        ):
            return True
        return False

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        poly = structure[idx_poly_with_error]
        if domain.geometry.is_closed and domain.geometry.is_convex and (poly[0] != poly[-1]):
            poly.points.append(poly.points[0])
        elif not domain.geometry.is_closed and (poly[0] == poly[-1]):
            poly.points = poly.points[:-1]
        return poly


class PolygonNotOutOfBounds(PolygonRule):
    @staticmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        geom_poly_allowed = ShapelyPolygon(
            [ShapelyPoint(pt.x, pt.y) for pt in domain.allowed_area],
        )
        for pt in structure[idx_poly_with_error]:
            geom_pt = ShapelyPoint(pt.x, pt.y)
            if (
                not geom_poly_allowed.contains(geom_pt)
                and not geom_poly_allowed.distance(geom_pt) < domain.min_dist_from_boundary
            ):
                return False

        return True

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        point_moved = False
        poly = structure[idx_poly_with_error]
        for p_id, point in enumerate(poly):
            if point in domain.fixed_points:
                continue
            point.x = max(point.x, domain.min_x + domain.len_x * 0.05)
            point.y = max(point.y, domain.min_y + domain.len_y * 0.05)
            point.x = min(point.x, domain.max_x + domain.len_x * 0.05)
            point.y = min(point.y, domain.max_y + domain.len_y * 0.05)
            if point not in domain:
                new_point = domain.geometry.nearest_point(point, domain.bound_poly)
                poly.points[p_id] = new_point
                point_moved = True

        if point_moved:
            poly = domain.geometry.resize_poly(poly=poly, x_scale=0.8, y_scale=0.8)

        if poly[0] != poly[-1] and domain.geometry.is_closed:
            poly.points.append(poly[0])
        elif poly[0] == poly[-1] and not domain.geometry.is_closed:
            poly.points = poly.points[:-1]

        return poly


class PolygonNotSelfIntersects(PolygonRule):
    @staticmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        poly = structure[idx_poly_with_error]
        return not (
            len(poly) > 2
            and _forbidden_validity(
                explain_validity(
                    ShapelyPolygon([ShapelyPoint(pt.x, pt.y) for pt in poly]).boundary,
                ),
            )
        )

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        poly = structure[idx_poly_with_error]
        poly = domain.geometry.get_convex(poly)
        return poly


def _forbidden_validity(validity):
    return validity != 'Valid Geometry' and 'Ring Self-intersection' not in validity


class Rules(Enum):
    not_too_close_polygons = PolygonsNotTooClose()
    valid_polygon_geom = PolygonGeometryIsValid()
    not_out_of_bounds = PolygonNotOutOfBounds()
    not_self_intersects = PolygonNotSelfIntersects()
    not_overlaps_prohibited = PolygonNotOverlapsProhibited()
    not_too_close_points = PointsNotTooClose()
