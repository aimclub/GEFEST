import json
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from gefest.core.geometry import Structure


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def parse_structs(path: str) -> list[Structure]:
    """Generates list of structures from file with dicts.

    Args:
        path (str): Path to file contains Structure object dumps,
            splitted with new line.

    Returns:
        list[Structure]: Serialized population.

    """
    with open(path, 'r') as file:
        dict_strings = file.readlines()

    pop = []
    for dict_string in dict_strings:
        ind = Structure(**json.loads(dict_string))
        pop.append(ind)

    return pop


@logger.catch
def where(
    sequence: list[Any],
    mask_rule: Callable[[Callable], bool],
) -> list[int]:
    """Finds indexes of values in a sequence satisfying the mask_rule.

    Args:
        sequence (list[Any]): list of values
        mask_rule (Callable[[Callable], bool]): rule for selecting values
            (e.g. lambda val: val is not None)

    Returns:
        list[int]: List of indexes of elements satisfying the condition.

    """
    return [idx for idx, ind in enumerate(sequence) if mask_rule(ind)]
