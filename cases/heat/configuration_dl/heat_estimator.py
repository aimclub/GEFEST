from pathlib import Path

from gefest.tools.estimators.DL.heat.heat_cnn import HeatCNN
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator(path_to_cnn=False):
    """
    ::TODO:: make abstract version for the configuration function and specific realizations
    (maybe it is possible to name it like configurator class)
    """
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    if not path_to_cnn:
        root_path = Path(__file__).parent.parent.parent.parent
        path_to_cnn = f'{root_path}/gefest/tools/estimators/DL/heat/effnet_mean'

    cnn = HeatCNN(path_to_cnn)

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=cnn)

    return estimator
