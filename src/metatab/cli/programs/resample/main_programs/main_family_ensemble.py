import sys
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.prediction.dataframe import PredictionDataframe
from metatab.ensemble.family import FamilyEnsembleEstimator
from metatab.ensemble.utils import BagCV

from metatab.cli.programs.resample.helper import (
    pick_splitter,
    get_repetition_fold,
    log_iteration,
    populate_dict_lists_,
    get_resample_iteration_signature
)

from metatab.cli.helper import (
    create_logger,
    create_json_configuration_file,
    check_target_feature,
    check_holdout_train_size,
    adjust_io_paths_,
    manage_output_path,
    get_ensemble_configuration,
    download_required_surrogate_models,
    add_predict_attrs_to_estimator
)




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
    name_dataset = dl.generic_dataset_name
    logger.debug(f"\nBuilding family-ensemble on {name_dataset}!")
    
    # encode y
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    create_json_configuration_file(pars, configuration_filepath)
    df_pred_results = PredictionDataframe()
    dict_results = defaultdict(list)
    list_dfs_ensemble_info = []
    filepath_df_ensemble_info = output_dir / "family_ensemble.txt"

    splitter = pick_splitter(pars)
    rng_ensemble = np.random.default_rng(pars["seed_estimator"])
    configuration = get_ensemble_configuration(pars["ensemble_configuration"])
    
    if not pars["disable_additional_txt_output"]: 
        txt_folder = output_dir / "additional_txt_info"
        txt_folder.mkdir(exist_ok=True)
    
    # this is to avoid the first download inside the fit call inflating times
    download_required_surrogate_models(configuration)


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
            "classes": le.classes_,
            "classes_counts": np.unique(y_train.to_numpy(), return_counts=True)[1],
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
            add_predict_attrs_to_estimator(ensemble, le, X_train, y_train, name_dataset)
            ens_filepath = output_dir / "estimators" / f"ensemble_{iter_signature}.pkl"
            ensemble.save(ens_filepath)
        else:
            ensemble.delete_models_from_disk()

        if not pars["disable_additional_txt_output"]:
            txt_folder_iter = txt_folder / f"iter_{iter_signature}"
            txt_folder_iter.mkdir(exist_ok=True)
            np.savetxt(txt_folder_iter / "predicted_probabilities.txt", pred_proba, delimiter="\t")
            np.savetxt(txt_folder_iter / "y_true.txt", y_test, fmt="%.1i", delimiter="\t")


    if not pars["disable_additional_txt_output"]:
        np.savetxt(txt_folder / "classes.txt", le.classes_, fmt="%.1000s", delimiter="\t")

    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)
    
    df_ensemble_info = pd.concat(list_dfs_ensemble_info, axis=0, ignore_index=True)
    df_ensemble_info.to_csv(filepath_df_ensemble_info, sep="\t", index=False)

    logger.debug(f"Outputs created at {output_dir}")