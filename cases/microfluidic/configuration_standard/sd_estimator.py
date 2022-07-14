from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator(domain):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------

    comsol = Comsol(
        path_to_mph='C:/Users/nano_user/PycharmProjects/rbc-traps-generative-design/setup/rbc-trap-setup.mph')

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=comsol)

    return estimator
