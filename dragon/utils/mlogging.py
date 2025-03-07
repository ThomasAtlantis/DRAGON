import logging
from rich.logging import RichHandler


class Logger:

    @staticmethod
    def build(logger_name: str, level="NOTSET"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt=f"[bold yellow1]:{logger_name}:[/bold yellow1] %(message)s",
            datefmt="[%X]"
        )
        handler = RichHandler(markup=True, rich_tracebacks=True)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
