from contextlib import nullcontext as no_exception
from pathlib import Path

import pytest

from gefest.core.configs.utils import load_config
from gefest.core.geometry import Point, Polygon, Structure
from gefest.tools.tuners.tuner import GolemTuner, TunerType

filepath = Path(__file__)
test_config = load_config(str(filepath.parent) + '/test_config.py')
ref_tuner_cfg = test_config.tuner_cfg
ref_objectives_list = test_config.objectives


struct_for_tune = Structure(
    [Polygon([Point(x, y) for x, y in [(15, 15), (15, 2), (2, 2), (2, 15), (15, 15)]])]
)
struct_for_tune.fitness = [test_config.objectives[0](struct_for_tune)]


@pytest.mark.parametrize(
    ', '.join(
        [
            'tuner_name',
            'expectation',
        ]
    ),
    [
        ('iopt', no_exception()),
        ('optuna', no_exception()),
        ('sequential', no_exception()),
        ('simulataneous', no_exception()),
        (None, pytest.raises(ValueError)),
    ],
)
def test_golem_tuners_wrap(tuner_name, expectation):
    """Check all available tuner types on single objective synthetic."""
    test_config.objectives = [ref_objectives_list[0]]  # sinngle objective
    if tuner_name:
        test_config.tuner_cfg = ref_tuner_cfg
        test_config.tuner_cfg.tuner_type = tuner_name
    else:
        test_config.tuner_cfg = None

    with expectation:
        tuner = GolemTuner(test_config)
        res = tuner.tune(struct_for_tune)[0]
        assert isinstance(res, Structure)
        assert res.fitness[0] <= struct_for_tune.fitness[0]
        assert isinstance(tuner.tuner, getattr(TunerType, tuner_name).value)


def test_multiobj_tuner():
    """Check optuna tuner on multiobjective synthetic."""
    test_config.tuner_cfg = ref_tuner_cfg
    test_config.objectives = ref_objectives_list
    test_config.tuner_cfg.tuner_type = 'optuna'
    tuner = GolemTuner(test_config)
    results = tuner.tune(struct_for_tune)
    assert all(isinstance(res, Structure) for res in results)
    assert all(len(res.fitness) == len(test_config.objectives) for res in results)
