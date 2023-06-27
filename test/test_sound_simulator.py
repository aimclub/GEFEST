import pytest
import pickle
import numpy as np
from gefest.tools.estimators.simulators.sound_wave.sound_interface import SoundSimulator
from cases.sound_waves.configuration import sound_domain


def upload_file(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
        f.close()
    return file


domain, _ = sound_domain.configurate_domain(
    poly_num=1,
    points_num=30,
    is_closed=True,
)

structure_path = "test/files/standart_structure.pickle"
spl_path = "test/files/standart_spl.pickle"

standart_spl = upload_file(spl_path)
standart_structure = upload_file(structure_path)

sound = SoundSimulator(domain)


def test_sumulator():
    current_spl = sound.estimate(standart_structure)

    assert (np.isclose(current_spl, standart_spl)).all()
