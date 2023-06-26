import numpy as np
import pickle

from gefest.core.structure.structure import Structure, get_random_structure
from gefest.tools.estimators.simulators.sound_wave.sound_interface import SoundSimulator, generate_map
from gefest.tools.estimators.estimator import Estimator



def configurate_estimator(domain: 'Domain', path_best_struct=None):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    sound = SoundSimulator(domain=domain)

    if path_best_struct is None:
        print("please, set up the best spl matrix into configuration")
        print("the best structure will be generated randomly")
        rnd_structure = get_random_structure(domain)
        best_spl = generate_map(domain, rnd_structure)
    else:
        with open(path_best_struct, "rb") as f:
            best_structure = pickle.load(f)
        best_spl = sound.estimate(best_structure)
        best_spl = np.nan_to_num(best_spl)

    # Loss for minimizing, it is optional function
    def loss(struct: Structure, estimator):

        spl = estimator.estimate(struct)
        spl = np.nan_to_num(spl)

        l_f = np.sum(np.abs(best_spl - spl))

        return l_f

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=sound,
                          loss=loss)

    return estimator
