from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure


def test_structure_to_string():
    # Create a structure for testing
    structure = Structure(polygons=[
        Polygon(points=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        Polygon(points=[Point(2, 2), Point(3, 2), Point(3, 3), Point(2, 3)])
    ])

    # Check the string representation of the structure
    result = str(structure)

    # Define the expected output
    expected = '\nPolygon 0, size 4:\nPoint 0: x=0, y=0; Point 1: x=1, ' \
               'y=0; Point 2: x=1, y=1; Point 3: x=0, y=1; \nPolygon 1, ' \
               'size 4:\nPoint 0: x=2, y=2; Point 1: x=3, y=2; Point 2: x=3, y=3; Point 3: x=2, y=3; '

    # Assert that the result matches the expected output
    assert result == expected
