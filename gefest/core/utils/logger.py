import datetime

from loguru import logger
from pydantic import RootModel

from gefest.core.geometry import Structure


class LogDispatcher:
    """Makes some logging stuff."""

    def __init__(self, log_dir: str = 'logs', run_name='new_run') -> None:
        self.timenow = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        self.log_dir = log_dir
        self.run_name = run_name

    def log_pop(self, pop: list[Structure], step: str):
        """Saves provided population to json."""
        inividual_log_handler = logger.add(
            f'{self.log_dir}/{self.run_name}_{self.timenow}/{step.zfill(5)}.log',
            level=4,
            format='{message}',
            filter=lambda record: record['level'].name in ['Level 4'],
        )
        for ind in pop:
            dump = RootModel[Structure](ind).model_dump_json(
                exclude={'polygons': {'__all__': {'points': {'__all__': {'coords'}}}}},
            )
            logger.log(4, dump)

        logger.remove(inividual_log_handler)
