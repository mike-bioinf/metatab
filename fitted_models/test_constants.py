'''
Here we define different sets of model parameters in respect to the official 
ones defined in "fit/constants.py" to speed up the fitting procedure.
'''


TEST_ES_XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 10,
    "eval_metric": "logloss",
    "early_stopping_rounds": 4,
    "verbose_eval": 0,
    "verbosity": 0
}


TEST_XGBCLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 10,
    "verbosity": 0
}


TEST_RANDOM_FOREST_CLASSIFIER_FIXED_PARAMS = {
    "n_estimators": 10
}
