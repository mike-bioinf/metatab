import optuna
from sklearn.ensemble import ExtraTreesClassifier


def _extra_trees_sampler_function(trial: optuna.Trial) -> dict:
    point = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_categorical("max_features", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_samples": trial.suggest_categorical("max_samples", [0.7, 0.8, 0.9, 1.0]),
        "min_impurity_decrease": trial.suggest_categorical("min_impurity_decrease", ["zero", "positive"]),
    }
    
    if point["min_impurity_decrease"] == "zero":
        point["min_impurity_decrease"] = 0.0
    else:
        point["min_impurity_decrease"] = trial.suggest_float("mid_positive", 1e-5, 1e-3, log=True)

    return point


class ExtraTreeSpec:
    classifier_class = ExtraTreesClassifier
    early_stop_on_validation_set = False
    random_state_parameter = "random_state"
    n_threads_parameter = "n_jobs"
    device_parameter = None
    main_device = "cpu"
    supported_devices = ["cpu"]
    default_preprocessing = "base"
    default_params = {}
    fixed_params = {"n_estimators": 1000}
    callbacks_on_params = None
    sampler_function = _extra_trees_sampler_function
    initialize_search_function = lambda: None
    params_as_object_columns_in_df_search = ["max_features"]