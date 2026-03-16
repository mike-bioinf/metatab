import optuna
from functools import partial
from xgboost import XGBClassifier
from metatab.utils.core import adjust_objective_logloss_and_num_classes, adjust_es_logloss_metric



def _xgb_sampler_function(trial: optuna.Trial) -> dict:
    point = {
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 5, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5, log=True),
        "gamma": trial.suggest_float("gamma", 0.001, 5, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 5, log=True),
        "subsample": trial.suggest_categorical("subsample", [0.8, 0.9, 1]),
        "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.6, 0.7, 0.8, 0.9, 1])
    }

    if point["grow_policy"] == "depthwise":
        point["tree_method"] = trial.suggest_categorical("tree_method_depthwise", ["exact", "hist"])
        point["max_depth"] = trial.suggest_int("max_depth", 1, 8)
        point["max_leaves"] = 0 # no limit
    else:
        point["tree_method"] = trial.suggest_categorical("tree_method_not_depthwise", ["approx", "hist"])
        point["max_depth"] = 0 # no limit
        point["max_leaves"] = trial.suggest_int("max_leaves", 2, 128, log=True)

    return point


class _BaseXGBSpec:
    classifier_class = XGBClassifier
    random_state_parameter = "random_state"
    n_threads_parameter = "n_jobs"
    device_parameter = None
    main_device = "cpu"
    supported_devices = ["cpu"]
    default_preprocessing = "base"
    sampler_function = _xgb_sampler_function
    initialize_search_function = lambda: None
    params_as_object_columns_in_df_search = None


class XGBSpec(_BaseXGBSpec):
    early_stop_on_validation_set = False
    default_params = {"verbosity": 0}
    fixed_params = {"n_estimators": 1000, "verbosity": 0}
    callbacks_on_params = [
        partial(adjust_objective_logloss_and_num_classes, framework="xgboost"),
    ]


class EsXGBSpec(_BaseXGBSpec):
    early_stop_on_validation_set = True
    default_params = {
        "n_estimators": 10000, # high number for early stop
        "eval_metric": "logloss_to_adjust",
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }
    fixed_params = {
        "n_estimators": 10000,
        "eval_metric": "logloss_to_adjust",
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }
    callbacks_on_params = [
        partial(adjust_objective_logloss_and_num_classes, framework="xgboost"),
        partial(adjust_es_logloss_metric, framework="xgboost"),
    ]