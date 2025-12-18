def compute_tune_spaces_score(
    auc1: float,
    auc2: float,
    runtime1: float,
    runtime2: float,
    lambda_coeff: float,
    bias: float
) -> float:
    '''
    Compute a quality score between two configurations based on AUC and runtime.
    This function compares exactly two configurations. 
    The score indicates how much better the first configuration (auc1 and runtime1) 
    is relative to the second (auc2 and runtime2):
    Positive score → the first configuration outperforms the second.
    Negative score → the second configuration outperforms the first.
    Zero score → both configurations perform equally well (a rare case).

    Score -> 100*(auc1 - auc2) + sign*lambda_coeff*[max(t1, t2)/min(t1, t2)] + bias

    Parameters:
        auc1 (float): AUC value registered by the first configuration.
        auc2 (float): AUC value registered by the second configuration.
        runtime1 (float): Runtime of the first configuration.
        runtime2 (float): Runtime of the second configuration.
        lambda_coeff (float): 
            Coefficient that weights the runtime term in the formula.
            It's purpose is to control the importance of the runtime term in respect to the auc one.
            Must be a number in [0, inf].
        bias (float):
            Values that is summed to the score. 
            It's purpose is to bias the score in favor on the first (positive bias) 
            or second configuration (negative bias).
            Must be a number in [-inf, inf].

    Returns:
        float: Returns the score.
    '''
    for number, err_string in zip(
        [auc1, auc2, runtime1, runtime2, lambda_coeff],
        ["auc1", "auc2", "runtime1", "runtime2", "lambda_coeff"] 
    ):
        if number < 0:
            raise ValueError(f"The '{err_string}' cannot be negative.")
    
    for runtime, err_string in zip(
        [runtime1, runtime2],
        ["runtime1", "runtime2"]
    ):
        if runtime == 0:
            raise ValueError(f"{err_string} is zero.")

    if runtime1 > runtime2:
        runtime_sign = -1
    elif runtime1 < runtime2:
        runtime_sign = +1
    else:
        # in case of tie
        runtime_sign = 0
    
    time_term = runtime_sign * lambda_coeff * max(runtime1, runtime2) / min(runtime1, runtime2)
    return 100*(auc1 - auc2) + time_term + bias
