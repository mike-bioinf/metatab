import sys
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict
from metatab_utils.data_loader import DataLoader
from metatab_utils.prediction import PredictionDataframe
from estimators.utils.pick import pick_estimator_class
from estimators.estimators import TunedEstimator
from estimators.utils.general import check_meta_tuning_options, check_y_is_integer_encoded
from metalearning.load import query_surrogate_framework

from cli.programs.resample.helper import (
    pick_splitter,
    create_json_configuration_file,
    get_repetition_fold,
    log_iteration,
    populate_dict_lists_,
    get_iteration_estimator_filepath,
    silent_nanmin
)

from cli.helper import (
    create_logger,
    check_target_feature,
    check_early_stop_parameters,
    check_holdout_train_size,
    adjust_io_paths_,
    manage_output_path,
    build_early_stop_configuration,
    build_tune_configuration,
    resolve_preprocessing_info
)



def main_tune(pars: dict):
    logger = create_logger(sys.stdout)

    check_target_feature(pars)
    check_early_stop_parameters(pars)
    check_holdout_train_size(pars)

    if pars["tune_algo"] == "meta":
        check_meta_tuning_options(
            pars["estimator"],
            pars["preprocessing"],
            pars["tune_space"]
        )

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    early_stop_conf = build_early_stop_configuration(pars)
    tune_conf = build_tune_configuration(pars)

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

    logger.debug(
        f"\nLaunching {pars["tune_algo"]} tuned {pars["estimator"]}" + 
        f" with {pars["tune_space"]} space on {name_dataset}!"
    )

    splitter = pick_splitter(pars)
    estimator_class = pick_estimator_class(pars["estimator"], pars["estimator_mode"])
    rng_estimator = np.random.default_rng(pars["seed_estimator"])

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    create_json_configuration_file(pars, configuration_filepath)
    dict_results = defaultdict(list)
    df_pred_results = PredictionDataframe()
    dict_hpo = defaultdict(list)
    hpo_filepath = output_dir / "hpo.txt"

    # this is to avoid the first download inside the fit call inflating times
    if pars["tune_algo"] == "meta":
        _ = query_surrogate_framework(pars["estimator"])
    
    
    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

        # we pass different seeds to maximize resample entropy
        estimator: TunedEstimator = estimator_class(
            # we need to resolve the preprocessing to extract the related info
            preprocessing=resolve_preprocessing_info(pars),
            seed=int(rng_estimator.integers(0, 2**32)),
            n_threads=pars["nthreads"],
            early_stop_configuration=early_stop_conf,
            tune_configuration=tune_conf
        )

        t = time()
        estimator.fit(X_train, y_train)
        fit_time = time() - t
        logger.debug("\t-Estimator fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")
        
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()
        best_hps = estimator.get_best_hps()
        refit_time = estimator.get_refit_time()
        search_losses = estimator.get_search_losses()
        best_loss = silent_nanmin(search_losses)
        search_losses_dict = {f"loss_{i}": value_loss for i, value_loss in enumerate(search_losses)}
    
        t = time()
        pred_proba = estimator.predict_proba(X_test)
        predict_time = time() - t
        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")

        iter_results = {
            "dataset": name_dataset,
            "predict_dataset": name_dataset,
            "estimator": pars["estimator"],
            "estimator_mode": pars["estimator_mode"],
            "tune_space": pars["tune_space"],
            "tune_algo": pars["tune_algo"],
            "tune_n_iter": pars["tune_n_iter"],
            "n_threads": pars["nthreads"],
            "preprocessing": pars["preprocessing"],
            "splitting_mode": pars["splitting_mode"],
            "repetition": repetition,
            "fold": fold,
            **fit_preprocessing_dict,
            "y_train": y_train,
            "y_test": y_test,
            "pred_proba": pred_proba,
            "fit_time": fit_time,
            "predict_time": predict_time
        }

        populate_dict_lists_(dict_results, **iter_results)
    
        populate_dict_lists_(
            dictionary=dict_hpo,
            dataset=name_dataset,
            estimator=pars["estimator"],
            preprocessing=pars["preprocessing"],
            algo=pars["tune_algo"],
            n_iter=pars["tune_n_iter"],
            n_cv_repeats=pars["tune_n_cv_repeats"],
            n_cv_folds=pars["tune_n_cv_folds"],
            splitting_mode=pars["splitting_mode"], 
            repetition=repetition,
            fold=fold,
            refit_time=refit_time,
            **best_hps,
            best_loss=best_loss,
            **search_losses_dict
            )

        if pars["save_estimators"]:
            estimator.save(get_iteration_estimator_filepath(pars, repetition, fold))


    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)
    
    pd.DataFrame(dict_hpo).to_csv(hpo_filepath, sep="\t", index=False)
    logger.debug(f"Outputs created at {output_dir}")