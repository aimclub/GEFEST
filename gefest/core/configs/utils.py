import inspect
from pathlib import Path
from typing import Optional

import yaml
from shapely.geometry.point import Point

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.configs.tuner_params import TunerParams
from gefest.core.geometry import Point as G_Point
from gefest.core.geometry import Polygon, PolyID, Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective


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
    root_path = Path.cwd()
    relative_path = Path(cfg_py_path).relative_to(root_path.parent.resolve())
    import_string = '.'.join(relative_path.with_suffix('').parts[1:])
    module_ = __import__(import_string, fromlist=[None])
    if cfg_yaml_path:
        user_metrics = []
        for name, obj in inspect.getmembers(module_):
            if inspect.isclass(obj):
                if issubclass(obj, Objective) and obj is not Objective:
                    user_metrics.append(name)

        if not user_metrics:
            raise ValueError(f'No Objective class has been found in {cfg_py_path}.')

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


def create_prohibited(
    points_radius,
    targets: Optional[list[list]] = None,
    fixed_points: Optional[list[list]] = None,
    fixed_area: Optional[list[list]] = None,
) -> Structure:
    """Creates of fixed, prohibited structures. Polygons cannot cross them.

    Args:
        points_radius (_type_): _description_
        targets (Optional[list[list]], optional): Fixed targets inside domain. Defaults to None.
        fixed_points (Optional[list[list]], optional): Fixed lines inside domain. Defaults to None.
        fixed_area (Optional[list[list]], optional): Fixed areas inside domain. Defaults to None.

    Returns:
        Structure: Compilation of inputs as GEFEST Structure.

    """
    prohibited_area = []
    if targets is not None:
        target_polygons = [
            list(Point(target).buffer(points_radius).exterior.coords) for target in targets
        ]
        target_points = [[G_Point(p[0], p[1]) for p in target] for target in target_polygons]
        poly_targets = [Polygon(id_=PolyID.PROH_TARG, points=points) for points in target_points]
        prohibited_area += poly_targets

    if fixed_points is not None:
        fix_points = [[G_Point(p[0], p[1]) for p in fixed] for fixed in fixed_points]
        poly_fixed = [Polygon(id_=PolyID.PROH_POLY, points=points) for points in fix_points]
        prohibited_area += poly_fixed

    if fixed_area is not None:
        fix_area = [[G_Point(p[0], p[1]) for p in fixed] for fixed in fixed_area]
        poly_area = [Polygon(id_=PolyID.PROH_AREA, points=points) for points in fix_area]
        prohibited_area += poly_area

    struct = Structure(polygons=prohibited_area)

    return struct
