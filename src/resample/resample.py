import os
import sys
import pandas as pd
import numpy as np
from estimators.types import Estimator
from utils.prediction import PredictionDataframe
from utils.data_loader import DataLoader
from utils.general import create_logger, check_y_is_integer_encoded

from utils.helper_params import (
    check_fit_args, 
    manage_output_path, 
    adjust_io_paths_
)

from resample.params import parse_args, adjust_splitting_specs_
from resample.constants import PRED_DATAFRAME_RESULTS_FIXED_COLUMNS

from resample.resample_helper import (
    get_repetition_fold,
    pick_splitter,
    log_iteration
)

from resample.save import (
    create_dict_hpo, 
    create_dict_results, 
    populate_dict_result_, 
    populate_dict_hpo_,  
    get_estimator_filepath, 
    get_additional_columns_results,
    create_json_configuration_file
)

## TODO: change function file location?
from fit.fit_helper import pick_estimator_class




def main():

    pars = vars(parse_args(sys.argv[1:]))
    check_fit_args(pars)

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
    logger.debug("Data loaded in memory!\n")

    splitter = pick_splitter(pars)
    estimator_class = pick_estimator_class(pars)
    rng_estimator = np.random.default_rng(pars["seed"])

    results_columns = (
        PRED_DATAFRAME_RESULTS_FIXED_COLUMNS + 
        get_additional_columns_results(pars["preprocessing"])
    )
    
    dict_results = create_dict_results(results_columns)
    dict_hpo = create_dict_hpo(pars)


    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

        # we pass a different seed to maximize resample entropy
        estimator: Estimator = estimator_class(
            preprocessing=pars["preprocessing"],
            seed=rng_estimator.integers(0, 2**32, dtype=np.uint32)
        )

        ## TODO: here we must implement an universal fit adapter
        ## when estimators with a different fit signature are implemented
        ## MAYBE TO DO IN ABSTRACT CLASS ??
        estimator.fit(X_train, y_train)
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()
        best_hps = estimator.get_best_hps()
        logger.debug("\t -Estimator fitted on input data.")

        ## TODO: an universal adapter is requested due to estimators 
        # having a different predict_proba signature
        pred_proba = estimator.predict_proba(X_test)

        populate_dict_result_(
            dict_results=dict_results,
            dataset=name_dataset,
            y_train=y_train,
            y_test=y_test,
            estimator=pars["estimator"],
            predict_dataset=name_dataset,
            splitting_mode=pars["splitting_mode"],
            preprocessing = pars["preprocessing"],
            repetition=repetition,
            fold=fold,
            pred_proba=pred_proba,
            **fit_preprocessing_dict
        )

        if pars["tune"]:
            populate_dict_hpo_(
                dict_hpo,
                splitting_mode=pars["splitting_mode"],
                preprocessing = pars["preprocessing"],
                repetition=repetition,
                fold=fold,
                **best_hps
            )
        
        if pars["save_estimators"]:
            estimator_filepath = get_estimator_filepath(pars, repetition, fold)
            estimator.save(estimator_filepath)
        
        logger.debug("\t-Inference completed.\n")
    

    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    hpo_filepath = output_dir / "hpo.txt"
    configuration_filepath = output_dir / "configuration.json"
    
    df_pred_results = PredictionDataframe()
    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)

    if pars["tune"]:
        pd.DataFrame(dict_hpo).to_csv(hpo_filepath, sep="\t", index=False)

    create_json_configuration_file(pars, configuration_filepath)
    logger.debug(f"Outputs created at {output_dir}.")




if __name__ == "__main__":
    main()