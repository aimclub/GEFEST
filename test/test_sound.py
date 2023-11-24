import numpy as np

from gefest.tools.estimators.simulators.sound_wave.sound_interface import (
    generate_random_map,
)


def test_random_map():
    """Test generator of random obstacles."""
    random_map = generate_random_map((42, 42), 111)
    assert isinstance(random_map, np.ndarray)
    assert random_map.shape == (42, 42, 1)
