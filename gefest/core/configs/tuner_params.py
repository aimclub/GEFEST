import inspect
from ctypes import Structure
from typing import Callable, Union

from hyperopt import hp
from pydantic import BaseModel, ConfigDict, field_validator

from gefest.tools.tuners import utils


class TunerParams(BaseModel):
    """Dataclass for GolemTuner parameters aggreagtion.

    Provides easy configuration for tuner
    with built-in validation and serialization.
    """

    tuner_type: str
    """Type of GOLEM tuners to use.
    Available: 'iopt', 'optuna', 'sequential', 'simulataneous'.
    For tuner details see:
        https://thegolem.readthedocs.io/en/latest/api/tuning.html
        https://fedot.readthedocs.io/en/latest/advanced/hyperparameters_tuning.html
    """

    n_steps_tune: int
    """Number of tuner steps."""

    tune_n_best: int = 1
    """Top tune_n_best structures from provided population to tune."""

    hyperopt_dist: Union[Callable, str] = hp.uniform
    """Random distribution function."""

    verbose: bool = True
    """GOLEM console info."""

    variacne_generator: Union[Callable[[Structure], list[float]], str] = utils.percent_edge_variance
    """The function for generating the search space includes intervals
    for each component of each point of each polygon in the provided structure.

    Output format should be spicific dict configuration. For details see:
        https://thegolem.readthedocs.io/en/latest/api/tuning.html
    """

    timeout_minutes: int = 60
    """GOLEM argument."""

    @field_validator('tuner_type')
    @classmethod
    def tuner_type_validate(cls, value):
        """Checks if specified tuner exists."""
        if isinstance(value, str):
            opt_names = ['iopt', 'optuna', 'sequential', 'simulataneous']
            if value in opt_names:
                return value
            else:
                raise ValueError(f'Invalid distribution name: {value}. Allowed names: {opt_names}')
        else:
            raise ValueError(f'Invalid argument: {value} of type {type(value)}.')

    @field_validator('hyperopt_dist')
    @classmethod
    def hyperopt_fun_validate(cls, value):
        """Checks if hyperopt distribution function exists."""
        if isinstance(value, str):
            r_ = inspect.getmembers(hp, inspect.isfunction)
            fun_names = [i[0] for i in r_]
            if value in fun_names:
                return getattr(hp, value)
            else:
                raise ValueError(f'Invalid distribution name: {value}. Allowed names: {fun_names}')
        elif isinstance(value, Callable):
            if value.__module__.split('.')[0] == hp.__name__.split('.')[0]:
                return value
        else:
            raise ValueError(f'Invalid argument: {value} of type {type(value)}.')

    @field_validator('variacne_generator')
    @classmethod
    def variacne_generator_fun_validate(cls, value):
        """Checks if specified variance generation function exists."""
        fun_names = ['percent_edge_variance']
        if isinstance(value, str):
            if value in fun_names:
                return getattr(utils, value)
        elif isinstance(value, Callable):
            return value
        else:
            raise ValueError(f'Invalid argument: {value} of type {type(value)}.')

    model_config = ConfigDict({'arbitrary_types_allowed': True})
