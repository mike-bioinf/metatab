import numpy as np
from typing import Literal



def compute_upper_confidence_bound(
    mean: np.ndarray, 
    uncertainty: np.ndarray, 
    k: float | Literal["infer"],
    mean_direction: Literal["higher_is_better", "lower_is_better"],
    n_points: int | None = None
) -> np.ndarray:
    '''
    Computes the upper confidence bound (UCB) as (+1/-1) * mean + k * uncertainty.
    The computed score can be think of as a measure of promisingness.
    Note that indipendently of the mean direction, the UCB is always an higher is better metric.

    Parameters:
        mean (np.ndarray): 
            Array of mean values, aka the predictions of the surrogate model
        
        uncertainty (np.ndarray): 
            Array of uncertainty values over the mean values.
        
        mean_direction (Literal["higher_is_better", "lower_is_better"]):
            Informs about the mean "direction".
            If lower_is_better then a -1 is multiplied to the mean,
            assuring the higher_is_better direction of the UCB score.

        k (float | Literal["infer"]): 
            Factor that control the trade off between mean and uncertainty,
            aka exploitation/exploration.
        
        n_points (int | None, optional): 
            Number of points that will be proposed.
            Info needed to infer k. Higher the value higher the k.
    
    Returns:
        np.ndarray: The UCB scores.
    '''
    if isinstance(k, str) and n_points is None:
        raise ValueError("To infer 'k' n_points must be provided (currently None).")
    
    if isinstance(k, [float, int]) and n_points is not None:
        raise ValueError("k is passed as a number with n_points not None. Ambiguous setting.")

    if mean_direction not in ["higher_is_better", "lower_is_better"]:
        raise ValueError(
            "mean_direction can assume only two possible values: 'higher_is_better' or 'lower_is_better'."
        )

    k = k if isinstance(k, [float | int]) else _infer_k_factor(n_points)
    mean = mean if mean_direction == "higher_is_better" else -1*mean
    return mean + k * uncertainty



def _infer_k_factor(n_points: int) -> float:
    '''
    We let the k factor to be proportional to "n_points",
    since when we propose multiple points we want them to be 
    more diverse and uncertain.
    '''
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    elif n_points == 1:
        # we take the best performance-wise ignoring the uncertainty
        k = 0
    elif n_points <= 5:
        k = 1.2
    else:
        k = 1.8
    
    return k