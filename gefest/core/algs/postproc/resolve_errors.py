import copy
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from enum import Enum
from itertools import combinations
from typing import Union

import numpy as np
from loguru import logger
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import explain_validity

from gefest.core.geometry import Polygon, Structure, get_random_poly
from gefest.core.geometry.domain import Domain
from gefest.core.viz.struct_vizualizer import GIFMaker


class PolygonRule(metaclass=ABCMeta):
    """Interface of postprocessing rule for polygon.
    Provides validation and correct functions for spicific error,
    e.g. 'out of bounds', 'self intersection', 'unclosed polygon'.
    """

    @staticmethod
    @abstractmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        """Check if there is no error in the spicific polygon in structure.

        Args:
            structure (Structure): Structure with error.
            idx_ (int): Index of polygon with error in structure.

        Returns:
            bool: True if polygon has no spicific problem,
                otherwise False.
        """
        ...

    @staticmethod
    @abstractmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        """Try to resolve spicific problem.
        The method does not guarantee error correction.

        Args:
            structure (Structure): Structure with error.
            idx_poly_with_error (int): Index of polygon with error in structure.

        Returns:
            Polygon
        """
        ...


class StructureRule(metaclass=ABCMeta):
    """Interface of postprocessing rule for whloe structure.
    Provides validation and correct functions for spicific error,
    e.g. 'polygons in structure too close'.
    """

    @staticmethod
    @abstractmethod
    def validate(
        structure: Structure,
        domain: Domain,
    ) -> bool:
        """Check if there is no error in the structure.

        Args:
            structure (Structure): Structure for validation.

        Returns:
            bool: True if structure has no spicific problem,
                otherwise False.
        """
        ...

    @staticmethod
    @abstractmethod
    def correct(
        structure: Structure,
        domain: Domain,
    ) -> Structure:
        """Try to resolve spicific problem.
        The method does not guarantee error correction.

        Args:
            structure (Structure): Structure with error.

        Returns:
            Structure
        """
        ...


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
        lenght = domain.dist_between_points
        check = []
        norms = []
        for pair in zip(structure[idx_poly_with_error][:-1], structure[idx_poly_with_error][1:]):
            norm = np.linalg.norm(np.array(pair[1].coords) - np.array(pair[0].coords), ord=1)
            norms.append(norm)
            check.append(norm > lenght)

        # logger.info(f'{norms}, {lenght}, {all(check)}')
        return all(check)

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        parent_structure = copy.deepcopy(structure)
        del parent_structure.polygons[idx_poly_with_error]
        poly = get_random_poly(parent_structure, domain)
        if poly is None:
            poly = structure[idx_poly_with_error]
        return poly


class PolygonNotClosed(PolygonRule):
    @staticmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        poly = structure[idx_poly_with_error]
        return domain.geometry.is_closed and (poly[0] == poly[-1])

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        poly = structure[idx_poly_with_error]
        if domain.geometry.is_closed and (poly[0] != poly[-1]):
            poly.points.append(poly.points[0])
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
                    ShapelyPolygon([ShapelyPoint(pt.x, pt.y) for pt in poly]),
                ),
            )
        )

    @staticmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        return domain.geometry.get_convex(structure[idx_poly_with_error])


def _forbidden_validity(validity):
    return validity != 'Valid Geometry' and 'Ring Self-intersection' not in validity


def validate(
    structure: Structure,
    rules: list[Union[StructureRule, PolygonRule]],
    domain: Domain,
) -> bool:
    if structure is None:
        return False
    if any(
        [(not poly or len(poly) == 0 or any([not p for p in poly])) for poly in structure],
    ):
        print('Wrong structure - problems with points')
        return False

    # for rule in (rule for rule in rules if isinstance(rule, PolygonRule)):
    #     if not all([rule.validate(structure, idx_, domain) for idx_, _ in enumerate(structure)]):
    #         return False
    # for rule in (rule for rule in rules if isinstance(rule, StructureRule)):
    #     if not rule.validate(structure, domain):
    #         return False

    for rule in (rule for rule in rules if isinstance(rule, PolygonRule)):
        for idx_, _ in enumerate(structure):
            if not rule.validate(structure, idx_, domain):
                return False

    for rule in (rule for rule in rules if isinstance(rule, StructureRule)):
        if not rule.validate(structure, domain):
            return False

    return True


from uuid import uuid4


def postprocess(
    structure: Structure,
    rules: list[Union[StructureRule, PolygonRule]],
    domain: Domain,
    attempts: int = 3,
) -> Union[Structure, None]:
    """Apply postprocessing rules to structure.

    Args:
        structure (Structure): Structure for postprocessing.
        rules (list[Union[StructureRule, PolygonRule]]): Postprocessing rules,
            which expect whole structure or particular polygon for check.
            This interfaces have check() and corerect() methods.
        domain (Domain): domain
        attempts (int, optional): Number of attempths to fix errors. Defaults to 3.

    Returns:
        Union[Structure, None]: If structure valid according to the rules,
            correct stucture will be returned, else None.

    """
    if structure is None:
        return None
    if any(
        [(not poly or len(poly) == 0 or any([not pt for pt in poly])) for poly in structure],
    ):
        print('Wrong structure - problems with points')
        return None

    corrected_structure = deepcopy(structure)
    gm = GIFMaker(domain=domain)
    colors = {
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red',
        4: 'purple',
    }

    gm.create_frame(corrected_structure, {'test': 'start'})
    for idx_, _ in enumerate(structure.polygons):
        for rule in (rule for rule in rules if isinstance(rule, PolygonRule)):
            for at in range(attempts):
                if not rule.validate(corrected_structure, idx_, domain):
                    corrected_structure[idx_] = rule.correct(corrected_structure, idx_, domain)
                    # gm.create_frame(corrected_structure, {f'{rule.__class__.__name__}, poly_id: {colors[idx_]}': f'attempt {at}'})
                else:
                    # gm.create_frame(corrected_structure, {f'{rule.__class__.__name__}, poly_id: {colors[idx_]}': 'OK'})
                    break
            else:
                if not rule.validate(corrected_structure, idx_, domain):
                    # gm.create_frame(corrected_structure, {f'{rule.__class__.__name__}, poly_id: {colors[idx_]}': 'Failed'})
                    # gm.make_gif(str(uuid4()))
                    return None

    for rule in (rule for rule in rules if isinstance(rule, StructureRule)):
        for at in range(attempts):
            if not rule.validate(corrected_structure, domain):
                corrected_structure = rule.correct(corrected_structure, domain)
                # gm.create_frame(corrected_structure, {f'{rule.__class__.__name__}, poly_id: {colors[idx_]}': f'attempt {at}'})
            else:
                # gm.create_frame(corrected_structure, {f'{rule.__class__.__name__}, poly_id: {colors[idx_]}': 'OK'})
                break
        else:
            if not rule.validate(corrected_structure, domain):
                # gm.create_frame(corrected_structure, {f'{rule.__class__.__name__}, poly_id: {colors[idx_]}': 'Failed'})
                # gm.make_gif(str(uuid4()))
                return None

    if validate(corrected_structure, rules, domain):
        return corrected_structure
    # gm.create_frame(corrected_structure, {f'Final validation : {rulename}': 'Failed'})
    # gm.make_gif(str(uuid4()))
    return None


class Rules(Enum):
    not_too_close_polygons = PolygonsNotTooClose()
    not_closed_polygon = PolygonNotClosed()
    not_out_of_bounds = PolygonNotOutOfBounds()
    not_self_intersects = PolygonNotSelfIntersects()
    not_too_close_points = PointsNotTooClose()
