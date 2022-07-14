from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain


# ------------
# USER-DEFINED CONFIGURATION OF DOMAIN FOR MICROFLUIDIC TASK
# ------------
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
    domain = Domain(allowed_area=[(-125, 100),
                                  (-75, 170),
                                  (15, 170),
                                  (30, 90),
                                  (-20, -130),
                                  (-20, -170),
                                  (-125, -170),
                                  (-125, 100)],
                    geometry=geometry,
                    max_poly_num=poly_num,
                    min_poly_num=1,
                    max_points_num=points_num,
                    min_points_num=min_points_num,
                    is_closed=is_closed)
    task_setup = Setup(domain=domain)

    return domain, task_setup
