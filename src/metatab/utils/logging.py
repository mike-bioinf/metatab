import logging
from typing import Literal


class FlushStreamHandler(logging.StreamHandler):
    '''
    A stream handler that flush when emits.
    Useful to deliver real time logging in HPC environment.
    '''
    def emit(self, record):
        super().emit(record)
        super().flush()


def create_logger(
    name: str, 
    stream,
    formatter: None | Literal["standard"] | logging.Formatter = None
) -> logging.Logger:
    '''
    Create a logger with a single handler to a stream with INFO level.
    If the logger already exists the function delete its handlers.

    Parameters:
        name (str): Logger name.
        stream: Either sys.stdout or sys.stderr.
        formatter (None | Literal["standard"]| Formatter, optional): 
            A Formatter object, None or "standard" in which case
            the classic time-stamp based formatter is used.
    
    Returns: The logger instance.
    '''
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    stream_handler = FlushStreamHandler(stream)
    stream_handler.setLevel(logging.DEBUG)
    if isinstance(formatter, str):
        formatter = logging.Formatter(
            fmt="[%(levelname).1s %(asctime)s,%(msecs)03d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger