from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain


def percent_edge_variance(
    structure: Structure,
    domain: Domain,
    percent: float = 0.5,
) -> list[float]:
    """Generates tuning variance for each point.

    Variance is equal to half the average edge length of the polygon.

    Returns:
        list[float]: list of variances for each point in structure
    """
    geom = domain.geometry
    variances = []
    for poly in structure:
        avg = percent * geom.get_length(poly) / (len(poly) - int(geom.is_closed))
        for point in poly:
            for coord in point.coords:
                variances.append((coord - avg, coord + avg))

    return variances
