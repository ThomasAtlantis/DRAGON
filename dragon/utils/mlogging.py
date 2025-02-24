import logging
from .singleton import SingletonType
from rich.logging import RichHandler


class Logger(metaclass=SingletonType):

    @staticmethod
    def build(logger_name: str, level="NOTSET"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(RichHandler())
        logger.propagate = False
        return logger
