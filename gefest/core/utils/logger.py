import datetime

from loguru import logger
from pydantic import RootModel

from gefest.core.geometry import Structure


class LogDispatcher:
    def __init__(self) -> None:
        self.timenow = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())

    def log_pop(self, pop: list[Structure], step: str):
        for ind_idx, ind in enumerate(pop):
            inividual_log_handler = logger.add(
                f'saved_pops/run_{self.timenow}/{step.zfill(5)}/{str((ind_idx+1)).zfill(5)}.log',
                level=4,
                format='{message}',
                filter=lambda record: record['level'].name in ['Level 4'],
            )
            logger.log(4, RootModel[Structure](ind).model_dump())
            logger.remove(inividual_log_handler)
        logger.info('Population logged. May be not sorted by fitness.')
