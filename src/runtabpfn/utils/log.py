import logging


def create_logger(stream) -> logging.Logger:
    '''
    Create a logger to a stream.
    Parameters:
        stream: Either sys.stdout or sys.stderr.
    Returns: The logger instance.
    '''
    logger = logging.getLogger("runtabpfn")
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
