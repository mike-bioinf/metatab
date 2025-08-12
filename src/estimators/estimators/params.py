from scipy.stats import randint, loguniform, uniform
from estimators.estimators.utils import int_loguniform



DEFAULT_TUNE_CONFIGURATION = {
    "configuration": "c0",
    "n_iter": 100,
    "n_repeats": 1,
    "n_splits": 5
}


SKLEARN_RANDOM_SEARCH_FIXED_PARAMS = {
    "scoring": "neg_log_loss",
    "n_jobs": 1,
    "refit": True
}



### XGBOOST ---------------------------------------------------------------------------------------

## We explore different quantization methods with different tree growing policy.
## We also consider more and less regularized configurations for most cases.
## With small sparse datasets the quanrization methods converge. This is True 
## especially for "hist" and "exact". Therefore we mainly explore them with
## different growing policy. We do not explore less regularized configuration
## for "approx" algo since it's slow.


XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 700,
    "verbosity": 0
}


ES_XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 10000,
    "eval_metric": "logloss_to_adjust",
    "early_stopping_rounds": 30, ## to increase? maybe 50?
    "verbose_eval": False,
    "verbosity": 0
}


RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS = {
    "grow_policy": ["depthwise"],
    "tree_method": ["exact"],
    "max_depth": randint(low=2, high=8),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": loguniform(a=0.001, b=5),
    "reg_alpha": loguniform(a=0.001, b=5),
    "gamma": loguniform(a=0.001, b=5),
    "min_child_weight": loguniform(a=0.001, b=5),
    "subsample": [0.8, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}


RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS_1 = {
    "grow_policy": ["lossguide"],
    "tree_method": ["approx"],
    "max_depth": [0],
    "max_leaves": int_loguniform(4, 128),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": loguniform(a=0.001, b=5),
    "reg_alpha": loguniform(a=0.001, b=5),
    "gamma": loguniform(a=0.001, b=5),
    "min_child_weight": loguniform(a=0.001, b=5),
    "subsample": [0.8, 0.9, 1],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1]
}


RANDOMIZED_XGBCLASSIFIER_PARAMS_DISTRIBUTIONS_2 = {
    "grow_policy": ["lossguide"],
    "tree_method": ["hist"],
    "max_depth": [0],
    "max_leaves": int_loguniform(4, 128),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": loguniform(a=0.001, b=5),
    "reg_alpha": loguniform(a=0.001, b=5),
    "gamma": loguniform(a=0.001, b=5),
    "min_child_weight": loguniform(a=0.001, b=5),
    "subsample": [0.8, 0.9, 1],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1]
}


## depthwise-exact with lower regularization
RANDOMIZED_XGB_CLASSIFIER_DISTRIBUTION_3 = {
    "grow_policy": ["depthwise"],
    "tree_method": ["exact"],
    "max_depth": randint(low=2, high=8),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": [[0], loguniform(a=0.001, b=5)], ## not working as is --> custom func/class scipy like???
    "reg_alpha": loguniform(a=0.001, b=5), ## 0_or_distr
    "gamma": loguniform(a=0.001, b=5),  ## 0_or_distr
    "min_child_weight": loguniform(a=0.001, b=5), ## 0_or_distr
    "subsample": [0.8, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}


## lossguide-hist with lower regularization
RANDOMIZED_XGB_CLASSIFIER_DISTRIBUTION_4 = {
    "grow_policy": ["lossguide"],
    "tree_method": ["hist"],
    "max_depth": randint(low=2, high=8),
    "learning_rate": loguniform(a=0.001, b=0.1),
    "reg_lambda": [[0], loguniform(a=0.001, b=5)], ## not working as is --> custom func/class scipy like???
    "reg_alpha": loguniform(a=0.001, b=5), ## 0_or_distr
    "gamma": loguniform(a=0.001, b=5), ## 0_or_distr
    "min_child_weight": loguniform(a=0.001, b=5), ## 0_or_distr
    "subsample": [0.8, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}




### CATBOOST ----------------------------------------------------------------------------------------

# We currently explore different variants only in terms of split quality score metrics.
# We keep the defaults when in comes to type of tree, leaf estimation method, 
# boosting type (no plain only ordered) and split finding algos.


## we list also the library defaults that we use just to be explicit
CATBOOST_FIXED_PARAMS = {
    "n_estimators": 1000,  ## more than xgboost since less prone to overfit
    "grow_policy": "SymmetricTree",
    "leaf_estimation_method": "Newton",
    "boosting_type": "Ordered",
    "feature_border_type": 'GreedyLogSum',
    "border_count": 254,
    "bootstrap_type": "Bayesian",
    "verbose": False
}


## we list also the library defaults that we use just to be explicit
ES_CATBOOST_FIXED_PARAMS = {
    "n_estimators": 10000,
    "eval_metric": "logloss_to_adjust",
    "grow_policy": "SymmetricTree",
    "leaf_estimation_method": "Newton",
    "boosting_type": "Ordered",
    "feature_border_type": 'GreedyLogSum',
    "border_count": 254,
    "bootstrap_type": "Bayesian",
    "early_stopping_rounds": 70,
    "od_type":"Iter", # classical early stop on validation set
    "use_best_model":True, # select early stopped ensemble
}


CATBOOST_C0 = {
    "score_function": ["Cosine"],
    "max_depth": list(range(2, 11)),
    "learning_rate": loguniform(0.001, 0.1),
    "leaf_estimation_iterations": int_loguniform(1, 10),
    "l2_leaf_reg": loguniform(1e-4, 5),
    "bagging_temperature": uniform(0, 1),
    "random_strength": list(range(1, 11)),
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}


CATBOOST_C1 = {
    "score_function": ["L2"],
    "max_depth": list(range(2, 11)),
    "learning_rate": loguniform(0.001, 0.1),
    "leaf_estimation_iterations": int_loguniform(1, 10),
    "l2_leaf_reg": loguniform(1e-4, 5),
    "bagging_temperature": uniform(0, 1),
    "random_strength": list(range(1, 11)),
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1]
}


## only on GPU
CATBOOST_C2 = {
    "score_function": ["NetwonCosine"],
    "max_depth": list(range(2, 11)),
    "learning_rate": loguniform(0.001, 0.1),
    "leaf_estimation_iterations": int_loguniform(1, 10),
    "l2_leaf_reg": loguniform(1e-4, 5),
    "bagging_temperature": uniform(0, 1)
}


## only on GPU
CATBOOST_C3 = {
    "score_function": ["NewtonL2"],
    "max_depth": list(range(2, 11)),
    "learning_rate": loguniform(0.001, 0.1),
    "leaf_estimation_iterations": int_loguniform(1, 10),
    "l2_leaf_reg": loguniform(1e-4, 5),
    "bagging_temperature": uniform(0, 1)
}




### RANDOM FOREST --------------------------------------------------------
RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 1000
}


RANDOMIZED_RANDOM_FOREST_PARAMS_DISTRIBUTIONS = {
    "max_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"],
    "min_samples_split": list(range(2, 21, 1)),
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_samples": [0.6, 0.7, 0.7, 0.9, 1]
}




### TABFPFN -----------------------------------------------------------
TABPFN_CLASSIFIER_FIXED_PARAMS = {
    "ignore_pretraining_limits": True,
    # suppressing categorical transformation 
    # that leads to testing data loss with sparse data
    "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0} 
}