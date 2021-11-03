from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.ops import nearest_points
from shapely.validation import explain_validity

from gefest.core.structure.polygon import Polygon
from gefest.core.utils import GlobalEnv

MIN_DIST = 15

min_dist_from_boundary = 1


def out_of_bound(structure: 'Structure', domain=None) -> bool:
    if domain is None:
        domain = GlobalEnv().domain
    geom_poly_allowed = GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in domain.allowed_area])

    for poly in structure.polygons:
        for pt in poly.points:
            geom_pt = GeomPoint(pt.x, pt.y)
            if not geom_poly_allowed.contains(geom_pt) and not \
                    geom_poly_allowed.distance(geom_pt) < min_dist_from_boundary:
                return True

    return False


def too_close(structure: 'Structure', domain) -> bool:
    is_too_close = any(
        [any([_pairwise_dist(poly_1, poly_2) < domain.min_dist for
              poly_2 in structure.polygons]) for poly_1
         in structure.polygons])
    return is_too_close


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon):
    if poly_1 is poly_2:
        return 9999

    nearest_pts = nearest_points(poly_1.as_geom(), poly_2.as_geom())
    return nearest_pts[0].distance(nearest_pts[1])


def self_intersection(structure: 'Structure'):
    return any([len(poly.points) > 2 and
                _forbidden_validity(explain_validity(GeomPolygon([GeomPoint(pt.x, pt.y) for pt in poly.points])))
                for poly in structure.polygons])


def _forbidden_validity(validity):
    return validity != 'Valid Geometry' and 'Ring Self-intersection' not in validity
