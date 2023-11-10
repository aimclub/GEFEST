from abc import ABCMeta, abstractmethod

from gefest.core.geometry import Polygon, Structure
from gefest.core.geometry.domain import Domain


class PolygonRule(metaclass=ABCMeta):
    """Interface of postprocessing rule for polygon.

    Provides validation and correction functions for spicific error,
    e.g. 'out of bounds', 'self intersection', 'unclosed polygon'.
    """

    @staticmethod
    @abstractmethod
    def validate(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> bool:
        """Checks if there is no error in the spicific polygon in structure.

        Args:
            structure (Structure): Structure with error.
            idx_ (int): Index of polygon with error in structure.

        Returns:
            bool: True if polygon has no spicific problem,
                otherwise False.
        """
        ...

    @staticmethod
    @abstractmethod
    def correct(
        structure: Structure,
        idx_poly_with_error: int,
        domain: Domain,
    ) -> Polygon:
        """Trys to fix spicific error.

        The method does not guarantee error correction.

        Args:
            structure (Structure): Structure with error.
            idx_poly_with_error (int): Index of polygon with error in structure.

        Returns:
            Polygon
        """
        ...


class StructureRule(metaclass=ABCMeta):
    """Interface of postprocessing rule for whole structure.

    Provides validation and correction functions for spicific error,
    e.g. 'polygons in structure too close'.
    """

    @staticmethod
    @abstractmethod
    def validate(
        structure: Structure,
        domain: Domain,
    ) -> bool:
        """Checks if there is no error in the structure.

        Args:
            structure (Structure): Structure for validation.

        Returns:
            bool: True if structure has no spicific problem,
                otherwise False.
        """
        ...

    @staticmethod
    @abstractmethod
    def correct(
        structure: Structure,
        domain: Domain,
    ) -> Structure:
        """Trys to fix spicific error.

        The method does not guarantee error correction.

        Args:
            structure (Structure): Structure with error.

        Returns:
            Structure
        """
        ...
