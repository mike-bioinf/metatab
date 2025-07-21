import numpy as np
from scipy.stats import randint, loguniform


# This number is used for all the randomized searches that we do,
# so both for the custom implementation and for sklearn searches.
N_ITERATIONS_RANDOM_SEARCH = 200


SKLEARN_RANDOM_SEARCH_FIXED_PARAMS = {
    "n_iter": N_ITERATIONS_RANDOM_SEARCH,
    "scoring": "neg_log_loss",
    "n_jobs": 1,
    "refit": True
}


XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 700,
    "n_jobs": -1,
    "verbosity": 0
}


ES_XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 10000,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "verbose_eval": False,
    "n_jobs": -1,
    "verbosity": 0
}


RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS = {
    "grow_policy": ["depthwise"],
    "tree_method": ["exact"],
    "max_depth": randint(low=2, high=6),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": loguniform(a=0.001, b=5),
    "reg_alpha": loguniform(a=0.001, b=5),
    "gamma": loguniform(a=0.001, b=5),
    "min_child_weight": loguniform(a=0.001, b=5),
    "subsample": [0.8, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}


RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 1000,
    "n_jobs": -1
}


RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS = {
    "max_features": np.linspace(0.1, 0.9, 9).round(1).tolist(),
    "min_samples_split": list(range(2, 21, 1)),
    "min_samples_leaf": list(range(1, 6, 1)),
    "max_samples": [0.6, 0.7, 0.7, 0.9, 1]
}


TABPFN_CLASSIFIER_FIXED_PARAMS = {
    "ignore_pretraining_limits": True,
    # suppressing categorical transformation 
    # that leads to testing data loss with sparse data
    "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0} 
}