import datetime

from loguru import logger
from pydantic import RootModel

from gefest.core.geometry import Structure


class LogDispatcher:
    def __init__(self, log_dir: str = 'logs', run_name='new_run') -> None:
        self.timenow = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        self.log_dir = log_dir
        self.run_name = run_name

    def log_pop(self, pop: list[Structure], step: str):
        inividual_log_handler = logger.add(
            f'{self.log_dir}/{self.run_name}_{self.timenow}/{step.zfill(5)}.log',
            level=4,
            format='{message}',
            filter=lambda record: record['level'].name in ['Level 4'],
        )
        for ind in pop:
            logger.log(4, RootModel[Structure](ind).model_dump())
        logger.remove(inividual_log_handler)
        logger.info('Population logged. May be not sorted by fitness.')
