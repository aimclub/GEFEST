from shapely.geometry import Polygon as shpoly
from gefest.core.geometry import Structure

def dice_metric(struct1: Structure, struct2: Structure):
    """
    Dice score for only one polygon in structure. This func calculate dice between two polygons.

    Args:
        stract1: Stacture with poly
        stract2: Stacture with poly

    Returns: [0;1] Coef how polygons similar to each other

    """

    try:
        p1 = shpoly([a.coords for a in [i.points for i in struct1.polygons][0]])
        p2 = shpoly([a.coords for a in [i.points for i in struct2.polygons][0]])
        dice = 2 * p1.intersection(p2).area / (p1.area + p2.area)
    except:
        dice = 0

    return dice
