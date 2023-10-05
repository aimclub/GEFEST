import datetime

from loguru import logger


class LogDispatcher:
    def __init__(self) -> None:
        timenow = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        logger.add(
            f'saved_pops/individuals_{timenow}.log',
            level=3,
            format='{message}',
            filter=lambda record: record['level'].name in ['Level 3', 'Level 4'],
        )
