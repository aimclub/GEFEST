import datetime

from loguru import logger
from pydantic import RootModel

from gefest.core.geometry import Structure


class LogDispatcher:
    def __init__(self) -> None:
        timenow = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        logger.add(
            f'saved_pops/individuals_{timenow}.log',
            level=3,
            format='{message}',
            filter=lambda record: record['level'].name in ['Level 3', 'Level 4'],
        )

    def log_pop(self, pop: list[Structure], step: int):
        logger.log(3, f'Step {step}')
        for ind in pop:
            logger.log(4, RootModel[Structure](ind).model_dump())
        logger.info('Population logged. May be not sorted by fitness.')
