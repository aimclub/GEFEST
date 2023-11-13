import pytest

from gefest.core.algs.postproc.rules import Rules
from gefest.core.geometry.geometry import Geometry
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.geometry.domain import Domain
from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.algs.geom.validation import *


import matplotlib.pyplot as plt
geometry = Geometry2D()
prohibited_area = [(30,30),
                   (30,50),
                   (50,50),
                   (50,30),
                   (30,30)]
prohibit_intersect = [(x,y-5) for x,y in prohibited_area]
prohibit_intersect_2 = [(x-5,y-5) for x,y in prohibited_area]
domain = Domain(allowed_area=[
                    [0, 0],
                    [0, 100],
                    [100, 100],
                    [100, 0],
                    [0, 0],
                ])
"""allowed area for Domain() is [(0, 0),
                                (0, 100),
                                (100, 100),
                                (100, 0)]
                        
"""
def plotc(coords):
    plt.plot([p[0] for p in coords],[p[1] for p in coords])


def poly_from_coords(coords):
    return Polygon(points=[Point(*coords) for coords in coords])


rules = Rules
poly_width = 10
poly_length = 20
# creating a testing polygons via corner points
rectangle_points = [
    (0, 0),
    (0, poly_length),
    (poly_width, poly_length),
    (poly_width, 0),
]
rectangle_poly = Polygon(points=[Point(*coords) for coords in rectangle_points]
)

triangle_points = [(0, 0), (poly_width, poly_length), (0, poly_length)]
triangle_poly = Polygon(points=[Point(*coords) for coords in triangle_points]
)

out_points = [Point(x + 200, y + 200) for (x, y) in rectangle_points]
out_poly = Polygon(points=out_points)

self_intersected_poly = [(2,2),(3,8),(2,1),(1,4),(9,9),(2,2)]
self_intersected_poly_open = [(2,2),(3,8),(2,1),(1,4),(9,9)]
self_intersected_poly_2 = [(4,4),(4,2),(2,2),(2,4),(4,4),(2,7),(4,7),(2,4)]
self_intersected_poly_3 = [(4,4),(4,2),(2,2),(2,4),(4,4),(2,7),(4,7),(2,4),(4,4),(4,2),(2,2),(2,4),(4,4)]
self_intersected_poly_4 = [(4,4),(4,2),(2,2),(2,4),(4,4),(4,2),(2,2),(2,4),(4,4)]
not_self_intersected_poly_closed = [(4,4),(4,2),(2,2),(2,4),(4,4)]
not_self_intersected_poly_open = [(4,4),(4,2),(2,2),(2,4)]
plotc([(1, 1),
        (1, poly_length),
        (poly_width, poly_length),
        (poly_width, 1)])
plotc(rectangle_points)
plotc(prohibited_area)
plotc(prohibit_intersect_2)
#plotc(self_intersected_poly_2)


structure_for_check = Structure(polygons=([poly_from_coords(coords) for coords in [self_intersected_poly,self_intersected_poly_open,not_self_intersected_poly_closed,not_self_intersected_poly_open]]))

rl_slf_inter = [rules.not_self_intersects.value.validate(structure_for_check,idx_poly_with_error=i,domain=domain) for i in range(len(structure_for_check.polygons))]
plt.show()
def test_intersection():
    intersected_points = [
        (1, 1),
        (1, poly_length),
        (poly_width, poly_length),
        (poly_width, 1),
    ]
    intersected_rectangle_poly = Polygon(points=[Point(*coords) for coords in intersected_points]
    )
    structure = Structure([rectangle_poly, intersected_rectangle_poly])
    assert intersection(structure, domain)

    structure = Structure([rectangle_poly, out_poly])
    assert not intersection(structure, domain)


def test_out_off_bound():
    inside_points = [
        (1, 1),
        (1, poly_length),
        (poly_width, poly_length),
        (poly_width, 1),
    ]
    inside_rectangle_poly = Polygon(points=[Point(*coords) for coords in inside_points]
    )
    structure = Structure([inside_rectangle_poly])
    assert not out_of_bound(structure, domain)

    structure = Structure([out_poly])
    assert out_of_bound(structure, domain)


def test_too_close():
    structure = Structure([rectangle_poly, triangle_poly])
    assert too_close(structure, domain)

    structure = Structure([rectangle_poly, out_poly])
    assert not too_close(structure, domain)


def test_self_intersection():
    incorrect_points = [
        (0, 0),
        (0, poly_length),
        (poly_width, poly_length),
        (poly_width - 5, poly_length - 5),
        (poly_width, poly_length + 10),
        (0, 0),
    ]
    incorrect_poly = Polygon(points=[Point(*coords) for coords in incorrect_points]
    )
    structure = Structure([incorrect_poly])
    assert self_intersection(structure)

    rebuilt_poly = geometry.get_convex(incorrect_poly)
    structure = Structure([rebuilt_poly])
    assert not self_intersection(structure)


def test_unclosed_poly():
    structure = Structure([rectangle_poly])
    assert unclosed_poly(structure, domain)

    closed_rectangle_points = [
        (0, 0),
        (0, poly_length),
        (poly_width, poly_length),
        (poly_width, 0),
        (0, 0),
    ]
    closed_rectangle_poly = Polygon(points=[Point(*coords) for coords in closed_rectangle_points]
    )
    structure = Structure([closed_rectangle_poly])
    assert not unclosed_poly(structure, domain)
