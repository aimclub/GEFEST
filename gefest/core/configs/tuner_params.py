from ctypes import Structure
from typing import Callable, Union
import inspect

from hyperopt import hp
from pydantic import BaseModel, ConfigDict, field_validator

from gefest.core.opt.tuning import utils


class TunerParams(BaseModel):
    """Dataclass for 'GolemTuner' parameters aggreagtion.
    Provides easy configuration for tuner
    with built-in validation and serialization.
    """

    tuner_type: str
    n_steps_tune: int
    hyperopt_dist: Union[Callable, str] = hp.uniform
    verbose: bool = True
    variacne_generator: Union[Callable[[Structure], list[float]], str] = utils.average_edge_variance
    timeout_minutes: int = 60

    @field_validator('tuner_type')
    def tuner_type_validate(cls, value):
        if isinstance(value, str):
            opt_names = ['iopt', 'optuna', 'sequential', 'simulataneous']
            if value in opt_names:
                return value
            else:
                raise ValueError(f'Invalid distribution name: {value}. Allowed names: {opt_names}')
        else:
            raise ValueError(f'Invalid argument: {value} of type {type(value)}.')

    @field_validator('hyperopt_dist')
    def hyperopt_fun_validate(cls, value):
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
    def variacne_generator_fun_validate(cls, value):
        fun_names = ["average_edge_variance"]
        if isinstance(value, str):
            if value in fun_names:
                return getattr(utils, value)
            else:
                raise ValueError(f'Invalid distribution name: {value}. Allowed names: {fun_names}')
        elif isinstance(value, Callable):
            if value.__module__.split('.')[0] == hp.__name__.split('.')[0]:
                return value
        else:
            raise ValueError(f'Invalid argument: {value} of type {type(value)}.')

    model_config = ConfigDict({'arbitrary_types_allowed': True})
