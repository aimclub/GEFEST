from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.validation import explain_validity

from gefest.core.structure.domain import Domain
from gefest.core.structure.polygon import Polygon

MIN_DIST = 15

min_dist_from_boundary = 1


def out_of_bound(structure: 'Structure', domain=None) -> bool:
    geom_poly_allowed = GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in domain.allowed_area])

    for poly in structure.polygons:
        for pt in poly.points:
            geom_pt = GeomPoint(pt.x, pt.y)
            if not geom_poly_allowed.contains(geom_pt) and not \
                    geom_poly_allowed.distance(geom_pt) < min_dist_from_boundary:
                return True

    return False


def too_close(structure: 'Structure', domain: Domain) -> bool:
    is_too_close = any(
        [any([_pairwise_dist(poly_1, poly_2, domain) < domain.min_dist for
              poly_2 in structure.polygons]) for poly_1
         in structure.polygons])
    return is_too_close


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    if poly_1 is poly_2 or len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    nearest_pts = domain.geometry.nearest_points(poly_1, poly_2)
    return domain.geometry.distance(nearest_pts[0], nearest_pts[1])


def self_intersection(structure: 'Structure'):
    return any([len(poly.points) > 2 and
                _forbidden_validity(explain_validity(GeomPolygon([GeomPoint(pt.x, pt.y) for pt in poly.points])))
                for poly in structure.polygons])


def unclosed_poly(structure: 'Structure') -> bool:
    return any([poly.points[0] != poly.points[-1] for poly in structure.polygons])


def _forbidden_validity(validity):
    return validity != 'Valid Geometry' and 'Ring Self-intersection' not in validity
