from pathlib import Path

from gefest.tools.estimators.simulators.comsol.comsol_interface import Comsol
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator(domain):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    root_path = Path(__file__).parent.parent.parent.parent
    comsol = Comsol(
        path_to_mph=f'{root_path}/gefest/tools/estimators/simulators/comsol/microfluid_file/rbc-trap-setup.mph')

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=comsol)

    return estimator
