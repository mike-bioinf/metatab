import logging


def create_logger(stream) -> logging.Logger:
    '''
    Create a logger to a stream at debug level.
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



def log_iteration(pars: dict, fold: int, repetition: int, logger: logging.Logger) -> None:
    '''Utility that logs info about the current iteration'''
    if pars["splitting_mode"] == "cv":
        logger.debug(f'Running on fold number {fold} of repetition number {repetition}:')
    elif pars["splitting_mode"] == "holdout":
        logger.debug(f'Running holdout iteration {fold}, with train size {pars["splitting_specs"]["train_size"]}:')