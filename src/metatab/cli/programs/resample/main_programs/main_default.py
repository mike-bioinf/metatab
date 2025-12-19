import sys
import numpy as np
from time import time
from collections import defaultdict
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.prediction import PredictionDataframe
from metatab.estimators.utils.pick import pick_estimator_class
from metatab.estimators.estimators import DefaultEstimator
from metatab.estimators.utils.general import check_y_is_integer_encoded

from metatab.cli.programs.resample.helper import (
    pick_splitter,
    get_repetition_fold,
    log_iteration,
    populate_dict_lists_,
    get_iteration_estimator_filepath
)

from metatab.cli.helper import (
    create_logger,
    create_json_configuration_file,
    check_target_feature,
    check_early_stop_parameters,
    check_holdout_train_size,
    adjust_io_paths_,
    manage_output_path,
    build_early_stop_configuration
)



def main_default(pars: dict):
    logger = create_logger(sys.stdout)

    check_target_feature(pars)
    check_early_stop_parameters(pars)
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

    logger.debug(f"\nLaunching {pars["estimator"]} on {name_dataset}!")

    splitter = pick_splitter(pars)
    early_stop_conf = build_early_stop_configuration(pars)
    estimator_class = pick_estimator_class(pars["estimator"], pars["estimator_mode"])
    rng_estimator = np.random.default_rng(pars["seed_estimator"])

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    create_json_configuration_file(pars, configuration_filepath)
    dict_results = defaultdict(list)
    df_pred_results = PredictionDataframe()

    
    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

        # we pass different seeds to maximize resample entropy
        estimator: DefaultEstimator = estimator_class(
            preprocessing=pars["preprocessing"],
            seed=int(rng_estimator.integers(0, 2**32)),
            n_threads=pars["nthreads"],
            early_stop_configuration=early_stop_conf
        )

        t = time()
        estimator.fit(X_train, y_train)
        fit_time = time() - t
        logger.debug("\t-Estimator fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")
        
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()
    
        t = time()
        pred_proba = estimator.predict_proba(X_test)
        predict_time = time() - t
        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")

        # store and/or save the iteration info
        iter_results = {
            "dataset": name_dataset,
            "predict_dataset": name_dataset,
            "estimator": pars["estimator"],
            "estimator_mode": pars["estimator_mode"],
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

        if pars["save_estimators"]:
            estimator.save(get_iteration_estimator_filepath(pars, repetition, fold))


    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)

    logger.debug(f"Outputs created at {output_dir}")