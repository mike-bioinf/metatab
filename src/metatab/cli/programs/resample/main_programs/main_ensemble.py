import sys
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.prediction.dataframe import PredictionDataframe
from metatab.estimators.utils.pick import pick_estimator_class
from metatab.estimators.estimators import EnsembledEstimator
from metatab.estimators.utils.general import check_meta_tuning_options
from metatab.metalearning.load import query_surrogate_framework

from metatab.cli.programs.resample.helper import (
    pick_splitter,
    get_repetition_fold,
    log_iteration,
    populate_dict_lists_,
    get_iteration_estimator_filepath,
    get_resample_iteration_signature
)

from metatab.cli.helper import (
    create_logger,
    create_json_configuration_file,
    check_target_feature,
    check_early_stop_parameters,
    check_device,
    check_holdout_train_size,
    adjust_io_paths_,
    manage_output_path,
    build_early_stop_configuration,
    build_ensemble_configuration,
    add_predict_attrs_to_estimator
)



def main_ensemble(pars: dict):
    logger = create_logger(sys.stdout)

    check_target_feature(pars)
    check_device(pars)
    check_early_stop_parameters(pars)
    check_holdout_train_size(pars)

    if pars["ensemble_algo"] == "meta":
        check_meta_tuning_options(
            pars["estimator"],
            pars["preprocessing"],
            pars["ensemble_space"]
        )

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)

    early_stop_conf = build_early_stop_configuration(pars)
    ens_conf = build_ensemble_configuration(pars)

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
    
    logger.debug(
        f"\nLaunching {pars["ensemble_algo"]} ensembled {pars["estimator"]}" + 
        f" with {pars["ensemble_space"]} space on {name_dataset}!"
    )

    # y encoding
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))

    splitter = pick_splitter(pars)
    estimator_class = pick_estimator_class(pars["estimator"], pars["estimator_mode"])
    rng_estimator = np.random.default_rng(pars["seed_estimator"])

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    create_json_configuration_file(pars, configuration_filepath)
    df_pred_results = PredictionDataframe()
    dict_results = defaultdict(list)
    list_dfs_ensemble_info = []
    filepath_df_ensemble_info = output_dir / "ensemble.txt"

    if not pars["disable_additional_txt_output"] and not pars["skip_inference"]: 
        txt_folder = output_dir / "additional_txt_info"
        txt_folder.mkdir(exist_ok=True)
    
    # this is to avoid the first download inside the fit call inflating times
    if pars["ensemble_algo"] == "meta" and not pars["ensemble_meta_surrogate_model"]:
        _ = query_surrogate_framework(pars["estimator"])
    

    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

        # we modify the ensemble configuration save_path to account for outer resample
        iter_folder_models = output_dir / "models" / f"iteration_{get_resample_iteration_signature(repetition, fold)}"
        ens_conf.save_path = iter_folder_models

        # we pass different seeds to maximize resample entropy
        estimator: EnsembledEstimator = estimator_class(
            preprocessing=pars["preprocessing"],
            seed=int(rng_estimator.integers(0, 2**32)),
            n_threads=pars["nthreads"],
            device=pars["device"],
            early_stop_configuration=early_stop_conf,
            ensemble_configuration=ens_conf
        )

        estimator.fit(X_train, y_train)
        fit_time = estimator.estimator_.fit_time_
        logger.debug("\t-Ensemble fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")
        
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()
        df_ensemble_members_recap = estimator.estimator_.df_members_
        df_ensemble_members_recap["dataset"] = name_dataset
        df_ensemble_members_recap["estimator"] = pars["estimator"]
        df_ensemble_members_recap["preprocessing"] = pars["preprocessing"]
        df_ensemble_members_recap["algo"] = pars["ensemble_algo"]
        df_ensemble_members_recap["n_members"] = pars["ensemble_n_members"]
        df_ensemble_members_recap["splitting_mode"] = pars["splitting_mode"]
        df_ensemble_members_recap["repetition"] = repetition
        df_ensemble_members_recap["fold"] = fold
        list_dfs_ensemble_info.append(df_ensemble_members_recap)

        if pars["save_estimators"]:
            add_predict_attrs_to_estimator(estimator, le, X_train, y_train, name_dataset)
            file = get_iteration_estimator_filepath(pars, repetition, fold)
            estimator.save(file)
            logger.debug(f"\t-Fitted model serialized at: {file}")
        else:
            estimator.estimator_.delete_models_from_disk()
            iter_folder_models.rmdir()

        if pars["skip_inference"]:
            logger.debug("\t-Skipped inference.\n")
            continue

        t = time()
        pred_proba = estimator.predict_proba(X_test)
        predict_time = time() - t
        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")

        iter_results = {
            "dataset": name_dataset,
            "predict_dataset": name_dataset,
            "estimator": pars["estimator"],
            "estimator_mode": pars["estimator_mode"],
            "ensemble_space": pars["ensemble_space"],
            "ensemble_algo": pars["ensemble_algo"],
            "ensemble_n_members": pars["ensemble_n_members"],
            "n_threads": pars["nthreads"],
            "preprocessing": pars["preprocessing"],
            "splitting_mode": pars["splitting_mode"],
            "repetition": repetition,
            "fold": fold,
            **fit_preprocessing_dict,
            "map_classes": str({c: i for c, i in enumerate(le.classes_)}),
            "classes": np.arange(le.classes_.size),
            "classes_counts": np.unique(y_train.to_numpy(), return_counts=True)[1],
            "y_test": y_test,
            "pred_proba": pred_proba,
            "fit_time": fit_time,
            "predict_time": predict_time
        }

        populate_dict_lists_(dict_results, **iter_results)

        if not pars["disable_additional_txt_output"]:
            txt_folder_iter = txt_folder / f"iter_{get_resample_iteration_signature(repetition, fold)}"
            txt_folder_iter.mkdir(exist_ok=True)
            np.savetxt(txt_folder_iter / "predicted_probabilities.txt", pred_proba, delimiter="\t")
            np.savetxt(txt_folder_iter / "y_true.txt", y_test, fmt="%.1i", delimiter="\t")


    # save ensemble info
    df_ensemble_info = pd.concat(list_dfs_ensemble_info, axis=0, ignore_index=True)
    df_ensemble_info.to_csv(filepath_df_ensemble_info, sep="\t", index=False)

    if not pars["save_estimators"]:
        (output_dir / "models").rmdir() 

    if pars["skip_inference"]:
        logger.debug(f"Outputs created at {output_dir}")
        return None

    if not pars["disable_additional_txt_output"]:
        np.savetxt(txt_folder / "classes.txt", le.classes_, fmt="%.1000s", delimiter="\t")

    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)

    logger.debug(f"Outputs created at {output_dir}")