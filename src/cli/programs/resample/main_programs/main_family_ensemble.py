import re
import sys
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict
from metatab_utils.data_loader import DataLoader
from metatab_utils.prediction import PredictionDataframe
from metalearning.load import query_surrogate_framework
from estimators.utils.general import check_y_is_integer_encoded
from ensemble.configuration import CollectionUserEnsembleConfiguration
from ensemble.family import FamilyEnsembleEstimator
from ensemble.utils import BagCV

from cli.programs.resample.helper import (
    pick_splitter,
    create_json_configuration_file,
    get_repetition_fold,
    log_iteration,
    populate_dict_lists_,
    get_resample_iteration_signature
)

from cli.helper import (
    create_logger,
    check_target_feature,
    check_holdout_train_size,
    adjust_io_paths_,
    manage_output_path
)




def get_ensemble_configuration(user_conf: str) -> CollectionUserEnsembleConfiguration:
    if re.match(r'^(all|cpu|gpu)_(meta|random)_\d+$', user_conf):
        return CollectionUserEnsembleConfiguration.create_predefined_collection(user_conf)
    else:
        return CollectionUserEnsembleConfiguration.load_json(user_conf)


def downaload_required_surrogate_models(collection: CollectionUserEnsembleConfiguration) -> None:
    '''Donload the surrogate models of the used meta-estimators'''
    [
        query_surrogate_framework(conf.estimator) 
        for conf in collection.configurations 
        if conf.algo == "meta"
    ]



def main_family_ensemble(pars: dict):
    logger = create_logger(sys.stdout)

    check_target_feature(pars)
    check_holdout_train_size(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)

    if pars["save_estimators"]:
        estimators_folder = pars["output_dir"] / "estimators"
        estimators_folder.mkdir(exist_ok=True)

    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y = dl.X, dl.y
    check_y_is_integer_encoded(y)
    name_dataset = dl.generic_dataset_name
    logger.debug(f"\nBuilding family-ensemble on {name_dataset}!")
    splitter = pick_splitter(pars)

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    create_json_configuration_file(pars, configuration_filepath)
    df_pred_results = PredictionDataframe()
    dict_results = defaultdict(list)
    list_dfs_ensemble_info = []
    filepath_df_ensemble_info = output_dir / "family_ensemble.txt"

    rng_ensemble = np.random.default_rng(pars["seed_estimator"])
    configuration = get_ensemble_configuration(pars["ensemble_configuration"])
    # this is to avoid the first download inside the fit call inflating times
    downaload_required_surrogate_models(configuration)


    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        iter_signature = get_resample_iteration_signature(repetition, fold)
        iter_seed = rng_ensemble.integers(0, 2**32)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

        # we pass different seeds to maximize resample entropy
        ensemble = FamilyEnsembleEstimator(
            name=pars["ensemble_name"],
            configuration=configuration,
            save_path=output_dir / "models" / f"iteration_{iter_signature}",
            bag_cv=BagCV(pars["bag_cv_repats"], pars["bag_cv_folds"], iter_seed) if pars["use_bag_cv"] else None,
            feature_space_ratio=pars["feature_space_randomization"],
            seed=iter_seed,
            time_limit=pars["ensemble_time_limit"],
            n_jobs=pars["nthreads"],
            log=50 #suppress logging
        )

        ensemble.fit(X_train, y_train)
        fit_time = ensemble.fit_time_
        df_ensemble_members_recap = ensemble.df_members_
        
        logger.debug("\t-Ensemble fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")
        
        t = time()
        pred_proba = ensemble.predict_proba(X_test)
        predict_time = time() - t
        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")

        iter_results = {
            "dataset": name_dataset,
            "predict_dataset": name_dataset,
            "estimator_mode": pars["estimator_mode"],
            "ensemble_configuration": pars["ensemble_configuration"],
            "use_bag_cv": pars["use_bag_cv"],
            "bag_cv_repeats": pars["bag_cv_repats"] if pars["use_bag_cv"] else None,
            "bag_cv_folds": pars["bag_cv_folds"] if pars["use_bag_cv"] else None,
            "n_threads": pars["nthreads"],
            "splitting_mode": pars["splitting_mode"],
            "repetition": repetition,
            "fold": fold,
            "y_train": y_train,
            "y_test": y_test,
            "pred_proba": pred_proba,
            "fit_time": fit_time,
            "predict_time": predict_time
        }

        populate_dict_lists_(dict_results, **iter_results)

        df_ensemble_members_recap["ensemble_configuration"] = pars["ensemble_configuration"]
        df_ensemble_members_recap["dataset"] = name_dataset
        df_ensemble_members_recap["splitting_mode"] = pars["splitting_mode"]
        df_ensemble_members_recap["repetition"] = repetition
        df_ensemble_members_recap["fold"] = fold
        list_dfs_ensemble_info.append(df_ensemble_members_recap)

        if pars["save_estimators"]:
            ens_filepath = output_dir / "estimators" / f"ensemble_{iter_signature}.pkl"
            ensemble.save(ens_filepath)
        else:
            ensemble.delete_models_from_disk()


    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)
    
    df_ensemble_info = pd.concat(list_dfs_ensemble_info, axis=0, ignore_index=True)
    df_ensemble_info.to_csv(filepath_df_ensemble_info, sep="\t", index=False)

    logger.debug(f"Outputs created at {output_dir}")