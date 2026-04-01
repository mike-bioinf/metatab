import optuna
from typing import Callable
from functools import partial
from lightgbm import LGBMClassifier
from metatab.utils.core import adjust_objective_logloss_and_num_classes, adjust_es_logloss_metric



def _lgbm_sampler_function(trial: optuna.Trial, prefix: str) -> dict:
    return {
        "learning_rate": trial.suggest_float(f"{prefix}__learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int(f"{prefix}__num_leaves", 2, 128, log=True),
        "max_bin": trial.suggest_categorical(f"{prefix}__max_bin", [5, 10, 20, 30, 50, 100, 150, 255]),
        "min_data_in_bin": trial.suggest_int(f"{prefix}__min_data_in_bin", 1, 5),
        "reg_alpha": trial.suggest_float(f"{prefix}__reg_alpha", 0.001, 5, log=True),
        "reg_lambda": trial.suggest_float(f"{prefix}__reg_lambda", 0.001, 5, log=True),
        "min_split_gain": trial.suggest_float(f"{prefix}__min_split_gain", 0.001, 5, log=True),
        "min_child_weight": trial.suggest_float(f"{prefix}__min_child_weight", 0.001, 5, log=True),
        "min_child_samples": trial.suggest_int(f"{prefix}__min_child_samples", 1, 5),
        "extra_trees": trial.suggest_categorical(f"{prefix}__extra_trees", [False, True]),
        "subsample": trial.suggest_categorical(f"{prefix}__subsample", [0.8, 0.9, 1]),
        "colsample_bytree": trial.suggest_categorical(f"{prefix}__colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }


class _BaseLGBMSpec:
    classifier_class = LGBMClassifier
    random_state_parameter = "random_state"
    n_threads_parameter = "n_jobs"
    device_parameter = None
    main_device = "cpu"
    supported_devices = ["cpu"]
    default_preprocessing = "base"
    set_params_function: Callable[[LGBMClassifier, dict], LGBMClassifier] = lambda cls, hps: cls.set_params(**hps)
    initialize_search_function = lambda: None
    params_as_object_columns_in_df_search = None


class LGBMSpec(_BaseLGBMSpec):
    type_classifier = "lgbm"
    early_stop_on_validation_set = False
    default_params = {
        "min_child_samples": 1, # the default of 31 is too high for small datasets
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True  # can speed up
    }
    fixed_params = {
        "n_estimators": 1000, # higher than default 100
        "boosting_type": "gbdt", # default, dart and rf alternatives
        "max_depth": -1, # default, no control
        "data_sample_strategy": "bagging", # default, more robust than goss
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True, # can speed up
        "subsample_freq": 1
    }
    hps_sampler_function = partial(_lgbm_sampler_function, prefix="lgbm")
    callbacks_on_params = [
        partial(adjust_objective_logloss_and_num_classes, framework="lightgbm"),
    ]



class EsLGBMSpec(_BaseLGBMSpec):
    type_classifier = "es_lgbm"
    early_stop_on_validation_set = True
    default_params = {
        "n_estimators": 10000, # high number for early stop
        "metric": "logloss_to_adjust",
        "early_stopping_rounds": 100,
        "min_child_samples": 1,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True
    }
    fixed_params = {
        "n_estimators": 10000,
        "early_stopping_rounds": 100,
        "boosting_type": "gbdt", 
        "max_depth": -1,
        "data_sample_strategy": "bagging",
        "verbose": -1,
        "early_stopping_min_delta": 0, # default, to avoid premature stopping
        "metric": "logloss_to_adjust",
        "deterministic": True,
        "force_col_wise": True,
        "subsample_freq": 1
    }
    hps_sampler_function = partial(_lgbm_sampler_function, prefix="es_lgbm")
    callbacks_on_params = [
        partial(adjust_objective_logloss_and_num_classes, framework="lightgbm"),
        partial(adjust_es_logloss_metric, framework="lightgbm"),
    ]