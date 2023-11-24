from contextlib import nullcontext as no_exception

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry2D

geometry = Geometry2D(is_convex=True, is_closed=True)
# marking length and width for testing polygon
poly_width = 10
poly_length = 20

# creating a testing polygons via corner points
rectangle_points = [
    (0, 0),
    (0, poly_length),
    (poly_width, poly_length),
    (poly_width, 0),
]
rectangle_poly = Polygon(points=[Point(*coords) for coords in rectangle_points])

triangle_points = [(0, 0), (poly_width, poly_length), (0, poly_length)]
triangle_poly = Polygon(points=[Point(*coords) for coords in triangle_points])

incorrect_points = [
    (0, 0),
    (0, poly_length),
    (poly_width, poly_length),
    (poly_width - 5, poly_length - 5),
    (-poly_width, -poly_length),
    (0, 0),
]
incorrect_poly = Polygon(points=[Point(*coords) for coords in incorrect_points])

# creating an expected rotated polygon for testing rotate_poly() function
exp_coords = [
    (-poly_width / 2, poly_width / 2),
    (-poly_width / 2, poly_length - poly_width / 2),
    (poly_length - poly_width / 2, poly_length - poly_width / 2),
    (poly_length - poly_width / 2, poly_width / 2),
]
exp_rectangle_poly = Polygon(points=[Point(*coords) for coords in exp_coords])


def test_resize_poly():
    """Test for resize_poly function from Geometry2D class."""
    x_scale = 2
    y_scale = 3

    original_poly = rectangle_poly
    resized_poly = geometry.resize_poly(original_poly, x_scale=x_scale, y_scale=y_scale)

    resized_square = geometry.get_square(resized_poly)
    original_square = geometry.get_square(original_poly)

    observed_difference = resized_square - original_square
    expected_difference = (
        (poly_width * x_scale) * (poly_length * y_scale)
    ) - poly_width * poly_length

    assert isinstance(resized_poly, Polygon)
    assert np.isclose(observed_difference, expected_difference)


@pytest.mark.parametrize('angle, expected_poly', [(90, exp_rectangle_poly), (180, rectangle_poly)])
def test_rotate_poly(angle, expected_poly):
    """Test for rotate_poly function from Geometry2D class."""
    rotate_poly = geometry.rotate_poly(rectangle_poly, angle=angle)

    rotated_coords = [tuple(coords.coords) for coords in rotate_poly.points]
    expected_coords = [tuple(coords.coords) for coords in expected_poly.points]

    assert set(rotated_coords).issubset(expected_coords) and len(rotated_coords) == len(
        expected_coords
    )


@pytest.mark.parametrize(
    'figure, expected_poly',
    [
        (rectangle_poly, poly_width * poly_length),
        (triangle_poly, poly_width * poly_length / 2),
        (Polygon([]), 0.0),
        (Polygon([Point(4, 33)]), 0.0),
        (Polygon([Point(0.234, 111), Point(42, 42)]), 0.0),
    ],
)
def test_get_square(figure, expected_poly):
    """Test for get_square function from Geometry2D class."""
    observed_square = geometry.get_square(figure)

    assert observed_square == expected_poly


@pytest.mark.parametrize('figure', [rectangle_poly, triangle_poly])
def test_contains_point(figure):
    """Test for get_square function from Geometry2D class."""
    expected_point = Point(1, 3)
    assert geometry.is_contain_point(figure, expected_point)

    expected_point = Point(-1, -1)
    assert not geometry.is_contain_point(figure, expected_point)


@pytest.mark.parametrize(
    'figure_1, figure_2, expected_point',
    [(Point(*rectangle_points[3]), rectangle_poly, Point(*rectangle_points[3]))],
)
def test_nearest_point(figure_1, figure_2, expected_point):
    """Test for nearest_point function from Geometry2D class."""
    observed_point = geometry.nearest_point(figure_1, figure_2)

    assert observed_point.coords == expected_point.coords


def test_intersects():
    """Test for intersects function from Geometry2D class."""
    assert geometry.intersects(Structure([rectangle_poly]))


def test_distance():
    """Test for distance function from Geometry2D class."""
    dist_1 = geometry.min_distance(rectangle_poly, triangle_poly)
    assert np.isclose(dist_1, 0)


def test_get_coords_from_linestring():
    """Test coordinate sequence extraction from LineString."""
    poly = LineString([(1, 2), (5, 6), (9, 3), (-5, 4)])
    coords = geometry.get_coords(poly)
    for idx_, point in enumerate(coords):
        assert isinstance(point, Point)
        assert poly.coords[idx_][0] == point.x and poly.coords[idx_][1] == point.y


def test_get_coords_from_polygon():
    """Test coordinate sequence extraction from Shapely polygon."""
    poly = ShapelyPolygon([(1, 2), (5, 6), (9, 3), (-5, 4)])
    coords = geometry.get_coords(poly)
    for idx_, point in enumerate(coords):
        assert isinstance(point, Point)
        assert poly.exterior.coords[idx_][0] == point.x and poly.exterior.coords[idx_][1] == point.y


@pytest.mark.parametrize(
    'points, angle',
    [
        ([11.005, -64.857, 89.07, -64.726, -44.4, 85.793, -98.744, -92.078], 107.085),
        ([52.247, -71.305, -24.941, -76.395, -77.306, -23.818, -84.436, 21.887], 84.906),
        ([-30.254, -50.595, -33.423, -17.237, 50.411, -53.9, -2.564, 10.087], 34.195),
        ([59.303, -20.546, 8.75, 67.479, 57.728, 90.01, 15.964, 30.346], 115.140),
        ([1.819, 1.014, 26.314, 1.47, 61.46, 40.183, 73.859, -52.156], 83.419),
        ([92.984, 6.033, -8.616, 86.894, -58.711, -25.121, 97.113, 6.373], 130.058),
        ([18.441, -27.842, 35.068, 97.597, 92.422, -73.318, -96.445, -61.18], 93.873),
        ([-32.867, 67.926, -26.673, 16.294, 66.072, 80.615, 42.02, 86.934], 111.561),
        ([43.228, 1.783, 77.274, 26.93, 36.935, 87.716, 29.722, 83.792], 172.097),
        ([-63.721, 56.527, 85.56, -3.208, 13.333, -84.431, -24.992, 88.125], 124.331),
        ([-76.41, -55.305, -9.261, 62.624, -94.735, -54.516, 92.331, 53.921], 30.243),
        ([65.439, 89.936, -76.353, -63.397, 41.992, 25.995, 54.238, 51.099], 163.243),
        ([-89.095, 5.421, 87.319, 39.346, -3.067, -24.067, -60.851, 20.656], 131.376),
        ([-12.762, 87.586, 76.559, -93.63, 75.78, -95.736, -0.895, -16.672], 162.117),
        ([-83.683, 63.124, 95.196, 63.648, -21.132, -37.336, -47.284, -49.189], 155.786),
        ([-36.463, -60.168, -39.688, -86.484, -70.261, -58.015, 79.795, -79.932], 88.677),
        ([-40.037, 82.116, -88.299, 43.377, -86.184, -57.963, 72.363, 56.635], 177.106),
        ([50.369, -50.69, 65.929, 70.203, 91.523, -30.051, -20.591, -40.274], 102.544),
        ([-39.418, 48.318, -93.891, -50.31, 0.702, -77.033, -87.242, 0.075], 102.332),
        ([-55.45, -44.895, 73.163, 3.177, -65.699, 59.205, -6.874, 72.387], 7.864),
        ([60.527, 48.915, 45.849, 54.572, -9.914, 81.24, -53.91, -82.924], 96.074),
        ([-89.957, 32.749, -25.919, 38.302, -72.819, 84.286, -54.435, -67.978], 88.072),
        ([-11.579, -86.434, -70.09, -98.124, -25.815, 65.467, 91.87, 58.349], 165.24),
        ([-50.353, -33.626, 2.378, 36.718, 70.838, 57.234, 77.679, 37.047], 124.424),
        ([-14.229, 72.8, -33.585, -47.179, 78.455, 48.313, -4.602, 85.821], 105.139),
        ([56.139, -75.325, -74.361, -72.867, -11.745, -35.636, 83.574, -57.986], 167.883),
        ([33.926, 77.767, 79.51, 10.045, -23.641, -6.021, -16.838, -25.08], 14.301),
        ([94.841, 59.284, -5.852, -43.094, 90.994, -68.019, 81.106, 25.351], 129.43),
        ([-23.61, 90.336, 75.407, 9.779, 5.59, -80.629, -87.663, 15.554], 173.245),
        ([-51.581, 21.225, 6.398, 97.242, -45.745, 30.03, -69.674, -57.325], 157.986),
    ],
)
def test_get_angle_between_two_vectors(points, angle):
    """Test angle evaluation between two segments."""
    v1 = (Point(points[0], points[1]), Point(points[2], points[3]))
    v2 = (Point(points[4], points[5]), Point(points[6], points[7]))
    assert round(geometry.get_angle(vector1=v1, vector2=v2), 3) == angle


@pytest.mark.parametrize(
    'poly, perimeter',
    [
        ([], 0.0),
        ([(-37.072, -23.071)], 0.0),
        ([(-37.072, -23.071), (-88.869, 6.951)], 0.0),
        (
            [
                (-37.072, -23.071),
                (-88.869, 6.951),
                (-92.754, 14.865),
                (-81.354, 86.284),
                (90.567, 85.27),
                (93.778, 32.003),
                (-15.131, -21.75),
                (-37.072, -23.071),
            ],
            509.728,
        ),
        ([(58.695, 30.38), (22.189, 69.546), (-3.151, 97.928), (58.695, 30.38)], 183.173),
        (
            [
                (-54.024, -33.25),
                (24.61, 53.479),
                (74.444, 99.953),
                (70.353, 8.636),
                (-54.024, -33.25),
            ],
            407.86,
        ),
        ([(63.546, -58.201), (-45.237, -45.838), (-94.31, -33.745), (63.546, -58.201)], 319.764),
        (
            [
                (-20.672, -95.466),
                (-96.978, -61.793),
                (-37.997, 85.886),
                (68.013, 21.485),
                (74.135, 5.526),
                (52.425, -69.268),
                (-20.672, -95.466),
            ],
            539.09,
        ),
        ([(79.756, -86.617), (-91.239, 5.298), (60.69, 81.567), (79.756, -86.617)], 533.393),
    ],
)
def test_get_length(poly, perimeter):
    """Test perimeter evaluation."""
    poly = Polygon([Point(p[0], p[1]) for p in poly])
    assert round(geometry.get_length(poly), 3) == perimeter


@pytest.mark.parametrize(
    'poly1, poly2, point',
    [
        (
            [
                [23.510440502339172, 87.18800203359015],
                [16.728428841295475, 78.83261611687797],
                [15.943128351745916, 71.8323686138699],
                [14.968358551861787, 60.697606629037665],
                [15.98053630885347, 56.32450382712932],
                [18.224675800169624, 51.74547631191035],
                [23.251098873883357, 47.709092938597536],
                [37.042664281727006, 41.305417493336904],
                [51.730349052022, 44.20922648597759],
                [55.34132820718901, 49.41624230136715],
                [60.850943092115024, 67.42906177406988],
                [60.151315546637846, 74.71046392129718],
                [59.518673819913424, 78.65796622807729],
                [52.07357384362214, 85.58361278820249],
                [23.510440502339172, 87.18800203359015],
            ],
            [
                [74.9022124262444, 93.0289081710956],
                [31.663987128692675, 91.89004979620276],
                [2.730841594823005, 86.30033381726267],
                [22.78157251716534, 17.337530029563908],
                [64.10004584470703, 5.75267485325562],
                [90.00707491266297, 19.79187044181077],
                [89.32384098619943, 44.0676925105978],
                [74.9022124262444, 93.0289081710956],
            ],
            (22.928, 90.202),
        ),
        (
            [
                [57.71183334020465, 87.67463234064934],
                [50.51452081685844, 85.13327478815634],
                [30.955034306867688, 78.09064628693557],
                [15.540522562298474, 58.754663565913],
                [1.2232768952884996, 37.53545119447447],
                [4.409631720280423, 32.084677628498895],
                [21.337745400878735, 14.100332298030487],
                [45.53144355498544, 5.9691262699021905],
                [69.27066967989211, 10.503485124834029],
                [78.81759801703856, 17.964969806084518],
                [82.92878296603564, 26.916540104922777],
                [78.69611307678483, 55.24020506910037],
                [70.40562083667376, 75.50919725166787],
                [57.71183334020465, 87.67463234064934],
            ],
            [
                [59.968260417247265, 79.11121326005193],
                [49.31168043482422, 74.20136803281687],
                [46.06820788459176, 71.40410908407036],
                [22.85222067774701, 47.932664979947454],
                [17.72673217438215, 39.19834540946594],
                [16.55881233231551, 13.981349428132969],
                [20.577748110574603, 6.1976506767562185],
                [21.974885499741816, 5.090091461486203],
                [50.24129554183121, 3.652798439526869],
                [56.637416610581845, 5.192537418677539],
                [69.00166666058895, 13.468678235144367],
                [83.46285749897223, 25.30437699123355],
                [92.01722715284058, 37.490366203765866],
                [91.66558358035604, 59.206524582221945],
                [66.76050969454376, 78.94647443873782],
                [59.968260417247265, 79.11121326005193],
            ],
            (16.788, 18.934),
        ),
        (
            [
                [41.57797516558374, 29.529792504473726],
                [39.45332651363959, 29.114683202717007],
                [34.84855968966326, 27.91300159267978],
                [32.88590087135793, 27.194306457226904],
                [32.92315979503104, 20.751070646641686],
                [33.43819137916439, 18.581875250584844],
                [41.811840139813384, 8.566101194161112],
                [45.496182969672034, 8.628727442113128],
                [50.096224072686724, 10.170585774637484],
                [53.849592181670545, 16.33796871836694],
                [53.06262872357211, 26.549357682842025],
                [45.6416971669762, 29.386565620239843],
                [41.57797516558374, 29.529792504473726],
            ],
            [
                [69.08902933506229, 93.01082354892503],
                [52.56687631956849, 90.81426107786706],
                [31.061085190433097, 82.26775080375413],
                [4.404951627468357, 38.84916757848918],
                [15.985917277009627, 12.435977235478646],
                [20.768028561626043, 7.772087054811841],
                [53.651812779362785, 3.2388064175488935],
                [83.71815758768406, 13.736787948448892],
                [93.12113335254287, 17.472924469847804],
                [94.17696875884448, 25.009223128368834],
                [93.03047974913059, 84.4360952349468],
                [87.631359798438, 91.02602964006971],
                [69.08902933506229, 93.01082354892503],
            ],
            (41.312, 4.94),
        ),
    ],
)
def test_nearest_points(poly1, poly2, point):
    """Test evaluation of neatest point to polygon in another polygon."""
    point_res = geometry.nearest_points(
        Polygon([Point(p[0], p[1]) for p in poly1]),
        Polygon([Point(p[0], p[1]) for p in poly2]),
    )
    assert (
        round(point_res.coords[0][0], 3) == point[0]
        and round(point_res.coords[0][1], 3) == point[1]
    )


@pytest.mark.parametrize(
    'poly, tolerance, expected',
    [
        (
            [
                (34.2, 16.2),
                (77.4, 16.4),
                (56.0, 30.2),
                (56.0, 55.0),
                (76.8, 75.8),
                (29.2, 75.4),
                (55.2, 55.4),
            ],
            0.5,
            [
                [34.158141791159785, 16.174805944079168],
                [55.16794940775614, 55.39311349505902],
                [29.12729011351953, 75.42438987524103],
                [76.86086771515441, 75.82551237609509],
                [56.025, 54.98964466094067],
                [56.025, 30.213625817351218],
                [77.48429325712596, 16.375389978643824],
                [34.158141791159785, 16.174805944079168],
            ],
        ),
        (
            [
                (34.2, 16.2),
                (77.4, 16.4),
                (56.0, 30.2),
                (56.0, 55.0),
                (76.8, 75.8),
                (29.2, 75.4),
                (55.2, 55.4),
            ],
            1,
            [
                [34.116283582319575, 16.149611888158326],
                [55.13589881551228, 55.386226990118026],
                [29.054580227039043, 75.44877975048206],
                [76.92173543030884, 75.8510247521902],
                [56.05, 54.97928932188134],
                [56.05, 30.227251634702437],
                [77.56858651425193, 16.350779957287646],
                [34.116283582319575, 16.149611888158326],
            ],
        ),
        (
            [
                (34.2, 16.2),
                (77.4, 16.4),
                (56.0, 30.2),
                (56.0, 55.0),
                (76.8, 75.8),
                (29.2, 75.4),
                (55.2, 55.4),
            ],
            10,
            [
                [34.2, 16.2],
                [77.4, 16.4],
                [56.0, 30.2],
                [56.0, 55.0],
                [76.8, 75.8],
                [29.2, 75.4],
                [55.2, 55.4],
                [34.2, 16.2],
            ],
        ),
        (
            [(21.0, 14.8), (79.2, 20.4), (43.8, 63.8)],
            0.5,
            [
                [20.958899610984233, 14.770929861582594],
                [43.79416105729854, 63.846711040065124],
                [79.24890909236552, 20.37959056769488],
                [20.958899610984233, 14.770929861582594],
            ],
        ),
        (
            [(21.0, 14.8), (79.2, 20.4), (43.8, 63.8)],
            1,
            [
                [20.917799221968462, 14.74185972316519],
                [43.788322114597094, 63.893422080130236],
                [79.29781818473103, 20.35918113538976],
                [20.917799221968462, 14.74185972316519],
            ],
        ),
        (
            [(21.0, 14.8), (79.2, 20.4), (43.8, 63.8)],
            10,
            [
                [20.177992219684604, 14.218597231651893],
                [43.68322114597095, 64.73422080130236],
                [80.17818184731024, 19.99181135389765],
                [20.177992219684604, 14.218597231651893],
            ],
        ),
        (
            [(25.2, 19.8), (81.4, 21.0), (25.8, 64.6), (76.8, 68.6)],
            0.5,
            [[25.2, 19.8], [25.8, 64.6], [76.8, 68.6], [81.4, 21.0], [25.2, 19.8]],
        ),
        (
            [(25.2, 19.8), (81.4, 21.0), (25.8, 64.6), (76.8, 68.6)],
            1,
            [[25.2, 19.8], [25.8, 64.6], [76.8, 68.6], [81.4, 21.0], [25.2, 19.8]],
        ),
        (
            [(25.2, 19.8), (81.4, 21.0), (25.8, 64.6), (76.8, 68.6)],
            10,
            [[25.2, 19.8], [25.8, 64.6], [76.8, 68.6], [81.4, 21.0], [25.2, 19.8]],
        ),
    ],
)
def test_simplify(poly, tolerance, expected):
    """Test polygon simplification."""
    res = geometry.simplify(
        Polygon([Point(p[0], p[1]) for p in poly]),
        tolerance,
    )
    assert [[p.x, p.y] for p in res] == expected


@pytest.mark.parametrize(
    'shapely_geom, expectation',
    [
        (ShapelyPolygon([(195, 46), (36, 128), (341, 250)]), no_exception()),
        (
            MultiPolygon(
                [
                    ShapelyPolygon([(195, 46), (36, 128), (341, 250)]),
                    ShapelyPolygon([(300, 458), (448, 455), (440, 310)]),
                ]
            ),
            no_exception(),
        ),
        (ShapelyPolygon(), pytest.raises(ValueError)),
        (MultiPolygon(), pytest.raises(ValueError)),
    ],
)
def test_get_random_point_in_shapey_geom(shapely_geom, expectation):
    """Test random point selection in arbitrary geometries."""
    with expectation:
        point = geometry.get_random_point_in_shapey_geom(shapely_geom)
        assert shapely_geom.contains(ShapelyPoint(point.coords))
