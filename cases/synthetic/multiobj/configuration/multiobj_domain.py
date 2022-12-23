from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain

X_domain_max = 300
X_domain_min = 0

Y_domain_max = 300
Y_domain_min = 0


def configurate_domain(poly_num: int,
                       points_num: int,
                       is_closed: bool):
    # ------------
    # GEFEST domain based on user-defined configuration_de
    # ------------
    if is_closed:
        min_points_num = 3
    else:
        min_points_num = 2

    geometry = Geometry2D(is_closed=is_closed)
    domain = Domain(allowed_area=[(X_domain_min, Y_domain_min),
                                  (X_domain_min, Y_domain_max),
                                  (X_domain_max, Y_domain_max),
                                  (X_domain_max, Y_domain_min),
                                  (X_domain_min, Y_domain_min)],
                    geometry=geometry,
                    max_poly_num=poly_num,
                    min_poly_num=1,
                    max_points_num=points_num,
                    min_points_num=min_points_num,
                    is_closed=is_closed)

    task_setup = Setup(domain=domain)

    return domain, task_setup