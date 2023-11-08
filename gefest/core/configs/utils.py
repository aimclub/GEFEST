import inspect
from pathlib import Path

import yaml

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective


class ParseMetricsError(Exception):
    def __init__(self, message=None, *args):
        if args:
            self.message = message

    def __str__(self):
        if self.message:
            return f'ParseMetricsError. {self.message}'


def load_config(
    cfg_py_path: str,
    cfg_yaml_path: str = None,
    *args,
    **kwargs,
) -> OptimizationParams:
    """Generates configuretion files from yaml files.

    Args:
        cfg_py_path (str): Path to metrics.py.
        cfg_yaml_path (str): Path to config.yaml.

    Returns:
        OptimizationParams: GEFEST unified configuretion file.
    """

    module_ = __import__(Path(cfg_py_path).stem)
    if cfg_yaml_path:
        user_metrics = []
        for name, obj in inspect.getmembers(module_):
            if inspect.isclass(obj):
                if issubclass(obj, Objective) and obj is not Objective:
                    user_metrics.append(name)

        if not user_metrics:
            raise ParseMetricsError('No Objective class has been loaded.')

        config_dict = yaml.safe_load(Path(cfg_yaml_path).read_text())
        domain_cfg = Domain.model_validate(config_dict['domain'])
        tuner_cfg = TunerParams.model_validate(config_dict['tuner_params'])

        config_dict['opt_params']['domain'] = domain_cfg
        config_dict['opt_params']['tuner_cfg'] = tuner_cfg
        config_dict['opt_params']['objectives'] = [
            getattr(module_, metric)(domain_cfg) for metric in user_metrics
        ]
        opt_params = OptimizationParams.model_validate(config_dict['opt_params'])
    else:
        opt_params = module_.opt_params
    return opt_params
