from pathlib import Path

from gefest.core.structure.structure import Structure
from gefest.tools.estimators.DL.bw_surrogate.bw_cnn import BWCNN
from gefest.tools.estimators.simulators.swan.swan_interface import Swan
from gefest.tools.estimators.estimator import Estimator
import cases.breakwaters.configuration_de.bw_domain as area


def configurate_estimator(domain, path_sim=False, path_sur=False):
    if not path_sim:
        root_path = Path(__file__).parent.parent.parent.parent
        path_sim = f"{root_path}/gefest/tools/estimators/simulators/swan/swan_model/"

    if not path_sur:
        root_path = Path(__file__).parent.parent.parent.parent
        path_sur = f"{root_path}/gefest/tools/estimators/DL/bw_surrogate/bw_surrogate_700_train.h5"

    swan = Swan(path=path_sim, targets=area.targets, grid=area.grid, domain=domain)

    cnn = BWCNN(domain=domain, path=path_sur, main_model=swan)

    # Multi-criteria loss function, in our case - HW and cost of structure
    def loss(struct: Structure, estimator):
        hs = estimator.estimate(struct)
        return hs

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=cnn, loss=loss)

    return estimator
