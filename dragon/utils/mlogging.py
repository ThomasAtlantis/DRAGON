import logging
from typing import Any, Dict
from rich.logging import RichHandler


class Logger:

    _configured_loggers: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def build(logger_name: str, level: str = "NOTSET") -> logging.Logger:
        logger = logging.getLogger(logger_name)
        if logger_name not in Logger._configured_loggers:
            logger.setLevel(level)
            formatter = logging.Formatter(
                fmt=f"[bold yellow1]:{logger_name}:[/bold yellow1] %(message)s",
                datefmt="[%X]"
            )
            handler = RichHandler(markup=True, rich_tracebacks=True, show_path=False)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

            Logger._configured_loggers[logger_name] = {
                "level": level,
                "formatter": formatter,
                "handler_type": type(handler)
            }
        return logger
