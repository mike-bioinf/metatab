import os
import sys
import pandas as pd
import numpy as np
from time import time
from estimators.types import Estimator
from utils.prediction import PredictionDataframe
from utils.data_loader import DataLoader
from utils.general import create_logger, check_y_is_integer_encoded

from utils.helper_params import (
    check_fit_args, 
    manage_output_path, 
    adjust_io_paths_,
    check_ambiguous_tune_setting
)

from resample.params import parse_args, adjust_splitting_specs_

from resample.resample_helper import (
    get_repetition_fold,
    pick_splitter,
    log_iteration,
    log_program_setting
)

from resample.save import (
    create_dict_hpo, 
    create_dict_results,  
    populate_dict_lists_,  
    get_estimator_filepath, 
    create_json_configuration_file
)

## TODO: change functions file location?
from fit.fit_helper import pick_estimator_class, pick_hps_configuration




def main():

    pars = vars(parse_args(sys.argv[1:]))
    check_fit_args(pars)
    check_ambiguous_tune_setting(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    adjust_splitting_specs_(pars)

    if pars["save_estimators"]:
        os.makedirs(pars["output_dir"] / "estimators", exist_ok=True)

    logger = create_logger(sys.stdout)
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y = dl.X, dl.y
    check_y_is_integer_encoded(y, is_predict_scenario=True)
    name_dataset = dl.generic_dataset_name

    log_program_setting(pars, logger, name_dataset)
    logger.debug("Data loaded in memory!\n")

    splitter = pick_splitter(pars)
    estimator_class = pick_estimator_class(pars)
    rng_estimator = np.random.default_rng(pars["seed"])
    hps_configuration = pick_hps_configuration(pars)

    dict_results = create_dict_results(pars)
    dict_hpo = create_dict_hpo(pars)


    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

        # we pass different seeds to maximize resample entropy
        estimator: Estimator = estimator_class(
            preprocessing=pars["preprocessing"],
            seed=int(rng_estimator.integers(0, 2**32)),
            params_distributions=hps_configuration
        )

        ## TODO: here we must implement an universal fit adapter
        ## when estimators with a different fit signature are implemented
        ## MAYBE TO DO IN ABSTRACT CLASS ??
        t = time()
        estimator.fit(X_train, y_train)
        fit_time = time() - t
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()
        best_hps = estimator.get_best_hps()
        logger.debug("\t-Estimator fitted on input data.")
        logger.debug(f"\t-Fit time in minutes (2-digits rounded): {round(fit_time/60, 2)}")

        ## TODO: an universal adapter is requested due to estimators 
        # having a different predict_proba signature

        t = time()
        pred_proba = estimator.predict_proba(X_test)
        predict_time = time() - t

        populate_dict_lists_(
            dictionary=dict_results,
            dataset=name_dataset, # must have the same length of the other inputs
            y_train=y_train,
            y_test=y_test,
            repetition=repetition,
            fold=fold,
            pred_proba=pred_proba,
            fit_time=fit_time,
            predict_time=predict_time,
            **fit_preprocessing_dict
        )

        if pars["tune"]:
            populate_dict_lists_(
                dictionary=dict_hpo,
                repetition=repetition,
                fold=fold,
                **best_hps
            )
        
        if pars["save_estimators"]:
            estimator_filepath = get_estimator_filepath(pars, repetition, fold)
            estimator.save(estimator_filepath)
        
        logger.debug(f"\t-Inference completed in minutes: {round(predict_time/60, 2)}\n")
    

    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    hpo_filepath = output_dir / "hpo.txt"
    configuration_filepath = output_dir / "configuration.json"
    
    df_pred_results = PredictionDataframe()
    
    df_pred_results.build_from_data(
        **dict_results, 
        save_path=output_dir,
        estimator=pars["estimator"],
        predict_dataset=name_dataset,
        splitting_mode=pars["splitting_mode"],
        preprocessing = pars["preprocessing"],
    )

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)

    if pars["tune"]:
        dict_hpo = {
            "splitting_mode": pars["splitting_mode"], 
            "preprocessing": pars["preprocessing"],
            **dict_hpo
        }
        pd.DataFrame(dict_hpo).to_csv(hpo_filepath, sep="\t", index=False)

    create_json_configuration_file(pars, configuration_filepath)
    logger.debug(f"Outputs created at {output_dir}")




if __name__ == "__main__":
    main()