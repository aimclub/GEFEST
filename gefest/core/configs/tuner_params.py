from ctypes import Structure
from typing import Callable, Union

from hyperopt import hp
from pydantic import BaseModel, ConfigDict, field_validator

from gefest.core.opt.tuning.uitls import average_edge_variance


class TunerParams(BaseModel):
    """Dataclass for 'GolemTuner' parameters aggreagtion.
    Provides easy configuration for tuner
    with built-in validation and serialization.
    """

    tuner_type: str
    n_steps_tune: int
    hyperopt_dist: Union[Callable, str] = hp.uniform
    verbose: bool = True
    variacne_generator: Callable[[Structure], list[float]] = average_edge_variance

    @field_validator('hyperopt_dist')
    def hyperopt_fun_validate(cls, value):
        if isinstance(value, str):
            return getattr(hp, value)
        elif isinstance(value, Callable):
            if value.__module__.split('.')[0] == hp.__name__.split('.')[0]:
                return value
        else:
            raise ValueError('Invalid argument')

    model_config = ConfigDict({'arbitrary_types_allowed': True})
