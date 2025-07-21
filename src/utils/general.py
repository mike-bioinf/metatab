import logging
import numpy as np
import pandas as pd



def check_y_is_integer_encoded(y: pd.Series, is_predict_scenario: bool = False) -> None:
    '''
    Checks that y is integer encoded. 
    This is essential to avoid errors in metrics computation.
    Raises different error messages depending on the scenario.
    '''
    y = np.asarray(y)
    
    if not np.issubdtype(y.dtype, np.integer):
        message = "Target variable y must be integer-encoded (e.g., 0, 1, 2, ...)."
        if is_predict_scenario:
            message += (
                " Note: in binary classification, class `1` is treated as the reference class"
                " in performance metrics computation."
            )
        raise ValueError(message)
    


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