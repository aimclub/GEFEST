from functools import reduce
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


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
        list[int]: _description_
    """
    return [idx for idx, ind in enumerate(sequence) if mask_rule(ind)]


def chained_call(*funcs):
    _initial_missing = object()

    def several_functions_chained_call(args=_initial_missing):
        if args is _initial_missing:
            return reduce(lambda r, f: f(r), funcs[1::], funcs[0]())
        return reduce(lambda r, f: f(r), funcs, args)

    def single_function_call(args=_initial_missing):
        if args is _initial_missing:
            return funcs[0]()
        return funcs[0](args)

    call = several_functions_chained_call if len(funcs) > 1 else single_function_call
    return call
