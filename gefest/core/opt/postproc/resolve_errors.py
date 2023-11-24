from copy import deepcopy
from typing import Union

from golem.utilities.data_structures import ensure_wrapped_in_sequence
from loguru import logger

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.postproc.rules_base import PolygonRule, StructureRule
from gefest.core.opt.postproc.validation import validate


class Postrocessor:
    """Implements logic of structures postprocessing."""

    @staticmethod
    def apply_postprocess(
        structures: Union[Structure, list[Structure]],
        rules: list[Union[StructureRule, PolygonRule]],
        domain: Domain,
        attempts: int = 3,
    ) -> list[Union[Structure, None]]:
        """Applys postprocessing rules over all provided structures."""
        structures = ensure_wrapped_in_sequence(structures)

        post_processed = [
            Postrocessor.postprocess_structure(struct, rules, domain, attempts)
            for struct in structures
        ]
        return post_processed

    @staticmethod
    def _apply_polygon_rule(structure, rule, attempts, domain) -> Union[Structure, None]:
        for idx_, _ in enumerate(structure.polygons):
            for _ in range(attempts):

                if not rule.validate(structure, idx_, domain):
                    structure[idx_] = rule.correct(structure, idx_, domain)
                else:
                    break
            else:
                if not rule.validate(structure, idx_, domain):
                    logger.info(f'{rule.__class__.__name__} fail')
                    return None

        return structure

    @staticmethod
    def _apply_structure_rule(structure, rule, attempts, domain) -> Union[Structure, None]:
        for _ in range(attempts):
            if not rule.validate(structure, domain):
                structure = rule.correct(structure, domain)
            else:
                break
        else:
            if not rule.validate(structure, domain):
                return None

        return structure

    @staticmethod
    def postprocess_structure(
        structure: Structure,
        rules: list[Union[StructureRule, PolygonRule]],
        domain: Domain,
        attempts: int = 3,
    ) -> Union[Structure, None]:
        """Apply postprocessing rules to structure.

        Args:
            structure (Structure): Structure for postprocessing.
            rules (list[Union[StructureRule, PolygonRule]]): Postprocessing rules,
                which expect whole structure or particular polygon for check.
                This interfaces have check() and corerect() methods.
            domain (Domain): domain
            attempts (int, optional): Number of attempths to fix errors. Defaults to 3.

        Returns:
            Union[Structure, None]: If structure valid according to the rules,
                correct stucture will be returned, else None.

        """
        if structure is None:
            logger.error('None struct postproc input')
            return None

        if any(
            (not poly or len(poly) == 0 or any(not pt for pt in poly))
            for poly in structure.polygons
        ):
            logger.error('Wrong structure - problems with points')
            return None

        corrected_structure = deepcopy(structure)

        for rule in (rule for rule in rules if isinstance(rule, PolygonRule)):
            corrected_structure = Postrocessor._apply_polygon_rule(
                corrected_structure,
                rule,
                attempts,
                domain,
            )
            if not corrected_structure:
                return None

        for rule in (rule for rule in rules if isinstance(rule, StructureRule)):
            corrected_structure = Postrocessor._apply_structure_rule(
                corrected_structure,
                rule,
                attempts,
                domain,
            )
            if not corrected_structure:
                return None

        if validate(corrected_structure, rules, domain):
            return corrected_structure

        logger.error('None struct postproc out')
        return None
