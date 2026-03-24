import optuna
from functools import partial
from catboost import CatBoostClassifier
from metatab.utils.core import adjust_objective_logloss_and_num_classes, adjust_es_logloss_metric



def _catboost_sampler_function(trial: optuna.Trial) -> dict:
    point = {
        "score_function": trial.suggest_categorical("score_function", ["Cosine", "L2"]),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "max_bin": trial.suggest_categorical("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_int("random_strength", 1, 11),
        "rsm": trial.suggest_categorical("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    if point["grow_policy"] != "SymmetricTree":
        point["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 5) # unique to one branch
    
    if point["grow_policy"] == "Lossguide":
        point["max_depth"] = 16
        point["max_leaves"] = trial.suggest_int("max_leaves", 2, 128, log=True) # unique to one branch
    else:
        point["max_depth"] = trial.suggest_int("max_depth", 1, 8)
    
    return point


class _BaseCatBoostSpec:
    classifier_class = CatBoostClassifier
    random_state_parameter = "random_state"
    n_threads_parameter = "n_threads"
    device_parameter = None
    main_device = "cpu"
    supported_devices = ["cpu"]
    default_preprocessing = "base"
    hps_sampler_function = _catboost_sampler_function
    initialize_search_function = lambda: None
    params_as_object_columns_in_df_search = None


class CatBoostSpec(_BaseCatBoostSpec):
    type_classifier = "catboost"
    early_stop_on_validation_set = False
    default_params = {"verbose": False, "allow_writing_files": False}
    fixed_params = {
        "n_estimators": 1000, # default
        "boosting_type": "Plain", 
        "leaf_estimation_method": "Newton", # default
        "feature_border_type": "GreedyLogSum", # default
        "bootstrap_type": "Bayesian", # default
        "verbose": False,
        "allow_writing_files": False,
    }
    callbacks_on_params = [partial(adjust_objective_logloss_and_num_classes, framework="catboost")]


class EsCatBoostSpec(_BaseCatBoostSpec):
    type_classifier = "es_catboost"
    early_stop_on_validation_set = True
    default_params = {
        "n_estimators": 10000,
        "early_stopping_rounds": 100,
        "od_type": "Iter", # classical early stop on validation set
        "eval_metric": "logloss_to_adjust",
        "use_best_model": True, # select early stopped ensemble
        "verbose": False,
        "allow_writing_files": False,
    }
    fixed_params = {
        "n_estimators": 10000,
        "early_stopping_rounds": 100,
        "boosting_type": "Plain",
        "eval_metric": "logloss_to_adjust",
        "od_type": "Iter",
        "use_best_model": True,
        "leaf_estimation_method": "Newton",
        "feature_border_type": "GreedyLogSum",
        "bootstrap_type": "Bayesian",
        "verbose": False,
        "allow_writing_files": False,
    }
    callbacks_on_params = [
        partial(adjust_objective_logloss_and_num_classes, framework="catboost"),
        partial(adjust_es_logloss_metric, framework="catboost"),
    ]