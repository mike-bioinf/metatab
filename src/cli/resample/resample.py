import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from copy import deepcopy
from time import time
from estimators import Estimator
from metatab_utils.prediction import PredictionDataframe
from metatab_utils.data_loader import DataLoader
from cli.resample.manager_estimator_workflow import GeneralManagerEstimatorWorkflowResample

from metatab_utils.helper_programs import (
    check_fit_resample_args,
    check_tune_configuration,
    manage_output_path, 
    adjust_io_paths_,
    adjust_tune_configuration_arg_,
    adjust_early_stopping_rounds_,
    pick_estimator_class,
    create_logger, 
    check_y_is_integer_encoded
)

from cli.resample.params import (
    parse_args,
    adjust_splitting_specs_
)

from cli.resample.helper import (
    get_repetition_fold,
    pick_splitter,
    log_iteration,
    log_program_setting,
    populate_dict_lists_,
    get_iteration_estimator_filepath, 
    create_json_configuration_file,
    silent_nanmin
)




## TODO: to clean the main program flow:
## abstract saving logic in a class
def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))
    check_fit_resample_args(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    adjust_splitting_specs_(pars)
    adjust_tune_configuration_arg_(pars)
    adjust_early_stopping_rounds_(pars)
    check_tune_configuration(pars, logger)

    if pars["save_estimators"]:
        os.makedirs(pars["output_dir"] / "estimators", exist_ok=True)

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

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    create_json_configuration_file(pars, configuration_filepath)
    
    dict_results = defaultdict(list)
    df_pred_results = PredictionDataframe()
    
    if pars["tune"]: 
        dict_hpo = defaultdict(list)
        hpo_filepath = output_dir / "hpo.txt"
        copy_tune_configuration = deepcopy(pars["tune_configuration"])
        del copy_tune_configuration["params_distributions"]

    
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
            n_threads=pars["nthreads"],
            early_stopping_rounds=pars["early_stopping_rounds"],
            tune_configuration=pars["tune_configuration"]
        )

        manager_estimator = GeneralManagerEstimatorWorkflowResample(estimator, pars, repetition, fold)
        manager_estimator.execute_pre_fit_routine()

        t = time()
        estimator.fit(X_train, y_train)
        fit_time = time() - t
        logger.debug("\t-Estimator fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")
        
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()
        manager_estimator.execute_post_fit_routine()

        if pars["tune"]:
            best_hps = estimator.get_best_hps()
            refit_time = estimator.get_refit_time()
            search_losses = estimator.get_search_losses()
            best_loss = silent_nanmin(search_losses)
            search_losses_dict = {f"loss_{i}": value_loss for i, value_loss in enumerate(search_losses)}
    
        t = time()
        pred_proba = estimator.predict_proba(X_test)
        predict_time = time() - t
        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")
        manager_estimator.execute_post_predict_routine()

        # store and/or save the iteration info
        iter_results = {
            "dataset": name_dataset,
            "predict_dataset": name_dataset,
            "estimator": pars["estimator"],
            "tune": pars["tune"],
            "tune_hps_configuration": pars["tune_configuration"]["configuration"] if pars["tune"] else None,
            "tune_algo": pars["tune_configuration"]["algo"] if pars["tune"] else None,
            "n_tune_iter": pars["tune_configuration"]["n_iter"] if pars["tune"] else None,
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

        if pars["tune"]:
            populate_dict_lists_(
                dictionary=dict_hpo,
                dataset=name_dataset,
                estimator=pars["estimator"],
                preprocessing=pars["preprocessing"],
                **copy_tune_configuration,
                splitting_mode=pars["splitting_mode"], 
                repetition=repetition,
                fold=fold,
                refit_time=refit_time,
                **best_hps,
                best_loss=best_loss,
                **search_losses_dict
            )

        if not pars["save_realtime"]:
            populate_dict_lists_(dict_results, **iter_results)
        else:
            df_pred_results.add_rows(
                iter_results, 
                compute_metrics=True,
                multiclass="average",
                average_strategy="macro" 
            )

            df_pred_results.to_csv(results_filepath, sep="\t", index=False)

            if pars["tune"]:
                pd.DataFrame(dict_hpo).to_csv(hpo_filepath, sep="\t", index=False)

        # save additional optional iteration-level info
        if pars["save_estimators"]:
            estimator.save(get_iteration_estimator_filepath(pars, repetition, fold))

    
    # Save final info when realtime mode is not enabled
    if not pars["save_realtime"]:
        df_pred_results.build_from_data(**dict_results, save_path=output_dir)

        if not df_pred_results.has_recovered:
            df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
            df_pred_results.to_csv(results_filepath, sep="\t", index=False)

        if pars["tune"]:
            pd.DataFrame(dict_hpo).to_csv(hpo_filepath, sep="\t", index=False)

    logger.debug(f"Outputs created at {output_dir}")




if __name__ == "__main__":
    main()