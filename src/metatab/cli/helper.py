import re
import json
import logging
from pathlib import Path
from typing import Literal
from metatab.estimators.params.utils import pick_estimator_tune_space
from metatab.estimators.utils.constants import EARLY_STOPPED_ESTIMATORS
from metatab.metalearning.load import query_surrogate_framework
from metatab.ensemble.configuration import CollectionUserEnsembleConfiguration

from metatab.metatab_utils.device import (
    check_cuda_is_available, 
    check_device_estimator_combination,
    resolve_device
)

from metatab.estimators.core.configurations import (
    EarlyStopConfiguration,
    TuneConfiguration,
    EnsembleConfiguration
)



def check_target_feature(pars: dict) -> None:
    '''Check that the target feature is set with df input-mode'''
    if pars["input_mode"] == "df" and pars["target_feature"] is None:
        raise ValueError("'--target-feature' must be specified when '--input-mode' equal 'df'.")


# def check_not_tunable_estimators(pars: dict, scenario: Literal["tune", "ensemble"]) -> None:
#     if (estimator := pars["estimator"]) in NON_TUNABLE_ESTIMATORS:
#         scenario_string = "tuned" if scenario == "tuned" else "ensembled"
#         raise ValueError(f"Estimator '{estimator}' cannot be {scenario_string}.")


# def check_incompatible_estimator_preprocessing(pars: dict) -> None:
#     if (
#         (estimator := pars["estimator"]) in PCA_INCOMPATIBLE_ESTIMATORS and 
#         pars["preprocessing"] == "pca"
#     ):
#         raise ValueError(f"PCA preprocessing cannot be used with '{estimator}' estimator.")


def check_early_stop_parameters(pars: dict) -> None:
    if pars["estimator"] in EARLY_STOPPED_ESTIMATORS:
        if pars["early_stop_rounds"] < 0 and pars["estimator"] not in ["realmlp", "tabm"]:
            raise ValueError("'early_stop_rounds' must be a >= 0.")
        if not 0 < pars["validation_set_size"] < 1:
            raise ValueError("'validation_set_size' must be a float in (0, 1).")


def check_holdout_train_size(pars: dict) -> None:
    if (
        pars["splitting_mode"] == "holdout" and 
        (pars["holdout_train_size"] <=0 or pars["holdout_train_size"] >= 1)
    ):
        raise ValueError(
            "'holdout_train_size' must be a float in (0, 1)."
        )


def check_device(pars: dict) -> None:
    '''General check on the device'''
    resolved_device = resolve_device(pars["device"], pars["estimator"])
    if resolved_device == "cuda": check_cuda_is_available()
    check_device_estimator_combination(resolved_device, pars["estimator"])


def adjust_io_paths_(pars: dict, input_arg: str, output_arg: str) -> None:
    '''
    Convert paths to absolute Path objects.
    The function works in place.
    '''
    pars[input_arg] = Path(pars[input_arg]).resolve()
    pars[output_arg] = Path(pars[output_arg]).resolve()


def adjust_paths_(pars: dict, *args) -> None:
    '''
    Convert the values associated to the key specified in `args`
    to absolute Path object. The function works in place.
    Is a version of `adjust_io_paths_` that works on multiple args
    '''
    for arg in args:
        pars[arg] = Path(pars[arg]).resolve()


def manage_output_path(pars: dict, output_arg: str, is_folder: bool) -> None:
    '''
    Control whether the output folder exists and whether to create it.
    One must specify the output parameter and if this is expected to be a folder. 
    If not the parent folder is considered.
    Assumes that the output argument is a Path object.
    '''
    out: Path = pars[output_arg]
    out_folder = out if is_folder else out.parent

    if not out_folder.exists() and not pars["create_outdir"]:
        raise FileNotFoundError(f"{out_folder} does not exists!")
    elif not out_folder.exists() and pars["create_outdir"]:
        out_folder.mkdir(parents=True, exist_ok=True)


def create_json_configuration_file(pars: dict, filepath: str | Path) -> None:
    '''Create a json representation of the input program configuration'''
    corrected_pars = {}
    # Path object cannot be serialized in json
    for k, v in pars.items():
        corrected_pars[k] = str(v) if isinstance(v, Path) else v    
    with open(filepath, "w") as f:
        json.dump(corrected_pars, f, indent=4)


def build_tune_configuration(pars: dict) -> TuneConfiguration:
    return TuneConfiguration(
        algo=pars["tune_algo"],
        n_iter=pars["tune_n_iter"],
        n_cv_repeats=pars["tune_n_cv_repeats"],
        n_cv_folds=pars["tune_n_cv_folds"],
        params_distributions=pick_estimator_tune_space(pars["estimator"], pars["tune_space"]),
        meta_surrogate_model=pars["tune_meta_surrogate_model"],
        meta_strategy=pars["tune_meta_strategy"]
    )


def build_ensemble_configuration(pars: dict) -> EnsembleConfiguration:
    return EnsembleConfiguration(
        name=pars["ensemble_name"],
        algo=pars["ensemble_algo"],
        n_members=pars["ensemble_n_members"],
        save_path=pars["output_dir"] / "models",
        params_distributions=pick_estimator_tune_space(pars["estimator"], pars["ensemble_space"]),
        meta_strategy=pars["ensemble_meta_strategy"],
        meta_surrogate_model=pars["ensemble_meta_surrogate_model"],
        time_limit=pars["ensemble_time_limit"],
        log=50 # we suppress logging
    )


def build_early_stop_configuration(pars: dict) -> None | EarlyStopConfiguration:
    if pars["estimator"] not in EARLY_STOPPED_ESTIMATORS:
        return None
    return EarlyStopConfiguration(
        early_stop_rounds=pars["early_stop_rounds"],
        validation_set_size=pars["validation_set_size"]
    )


def get_ensemble_configuration(user_conf: str) -> CollectionUserEnsembleConfiguration:
    '''
    Helper for family-ensemble scenario.
    Create the CollectionUserEnsembleConfiguration from user input.
    '''
    if re.match(r'^(all|cpu|gpu)_(meta|random)_\d+$', user_conf):
        return CollectionUserEnsembleConfiguration.create_predefined_collection(user_conf)
    else:
        return CollectionUserEnsembleConfiguration.load_json(user_conf)


def download_required_surrogate_models(collection: CollectionUserEnsembleConfiguration) -> None:
    '''
    Helper for family-ensemble scenario.
    Donwload the surrogate models of the requested meta-estimators.
    '''
    [
        query_surrogate_framework(conf.estimator)
        for conf in collection.configurations 
        if conf.algo == "meta"
    ]


class FlushStreamHandler(logging.StreamHandler):
    '''
    A stream handler that flush when emits.
    Useful to deliver real time logging in HPC environment.
    '''
    def emit(self, record):
        super().emit(record)
        super().flush()


def create_logger(stream) -> logging.Logger:
    '''
    Create a logger to a stream.
    Parameters:
        stream: Either sys.stdout or sys.stderr.
    Returns: The logger instance.
    '''
    logger = logging.getLogger("metatab")
    logger.setLevel(logging.DEBUG)
    stream_handler = FlushStreamHandler(stream)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger