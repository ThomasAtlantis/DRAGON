import logging
from .singleton import SingletonType
from rich.logging import RichHandler


class Logger(metaclass=SingletonType):

    @staticmethod
    def build(logger_name: str, level="NOTSET"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        formatter = logging.Formatter(datefmt="[%X]")
        handler = RichHandler(markup=True)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
