from gefest.tools.estimators.DL.heat.heat_cnn import CNN
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator():
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    cnn = CNN()

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=cnn)

    return estimator
