from typing import Union

from loguru import logger

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.postproc.rules_base import PolygonRule, StructureRule


def validate(
    structure: Structure,
    rules: list[Union[StructureRule, PolygonRule]],
    domain: Domain,
) -> bool:
    """Validates single structure.

    Args:
        structure (Structure): Structure.
        rules (list[Union[StructureRule, PolygonRule]]): Validation rules.
        domain (Domain): Task domain.

    Returns:
        bool: True if valid else False
    """
    if structure is None:
        return False

    if any((not poly or len(poly) == 0 or any(not p for p in poly)) for poly in structure):
        logger.error('Wrong structure - problems with points')
        return False

    for rule in (rule for rule in rules if isinstance(rule, PolygonRule)):
        for idx_, _ in enumerate(structure):
            if not rule.validate(structure, idx_, domain):
                logger.info(f'{rule.__class__.__name__} final fail')
                return False

    for rule in (rule for rule in rules if isinstance(rule, StructureRule)):
        if not rule.validate(structure, domain):
            logger.info(f'{rule.__class__.__name__} final fail')
            return False

    return True
