import logging


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
    formatter: logging.Formatter | None = None
) -> logging.Logger:
    '''
    Create a logger with a single handler to a stream with INFO level.
    If the logger already exists the function delete its handlers.

    Parameters:
        name (str): Logger name.
        stream: Either sys.stdout or sys.stderr.
        formatter (None | Formatter, optional): A Formatter object or None.
    
    Returns: The logger instance.
    '''
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    stream_handler = FlushStreamHandler(stream)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger