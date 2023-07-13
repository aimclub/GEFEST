import numpy as np
import pickle

from test.test_sound_simulator import load_file_from_path
from gefest.core.structure.structure import Structure, get_random_structure
from gefest.tools.estimators.simulators.sound_wave.sound_interface import (
    SoundSimulator,
    generate_map,
)
from gefest.tools.estimators.estimator import Estimator


def configurate_estimator(domain: "Domain", path_best_struct=None):
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
        best_structure = load_file_from_path(path_best_struct)
        best_spl = sound.estimate(best_structure)
        best_spl = np.nan_to_num(best_spl, nan=0, neginf=0, posinf=0)

    # Loss for minimizing, it is optional function
    def loss(struct: Structure, estimator):
        spl = estimator.estimate(struct)
        current_spl = np.nan_to_num(spl, nan=0, neginf=0, posinf=0)

        l_f = np.sum(np.abs(best_spl - current_spl))

        return l_f

    # ------------
    # GEFEST estimator
    # ------------

    # Here loss is an optional argument, otherwise estimator will be considered as loss for minimizing
    estimator = Estimator(estimator=sound, loss=loss)

    return estimator
