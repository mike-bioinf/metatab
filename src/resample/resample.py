import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from time import time
from estimators import Estimator
from metatab_utils.prediction import PredictionDataframe
from metatab_utils.data_loader import DataLoader

from metatab_utils.helper_programs import (
    check_fit_resample_args,
    check_tune_algo,
    manage_output_path, 
    adjust_io_paths_,
    adjust_tune_configuration_arg_,
    adjust_early_stopping_rounds_,
    pick_estimator_class,
    fix_estimator_fixed_params_during_resampling_,
    create_logger, 
    check_y_is_integer_encoded
)

from resample.params import (
    parse_args,
    adjust_splitting_specs_
)

from resample.resample_helper import (
    get_repetition_fold,
    pick_splitter,
    log_iteration,
    log_program_setting
)

from resample.save import ( 
    populate_dict_lists_,
    get_resample_iteration_signature,
    get_estimator_filepath, 
    create_json_configuration_file
)




def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_fit_resample_args(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    adjust_splitting_specs_(pars)
    adjust_tune_configuration_arg_(pars)
    adjust_early_stopping_rounds_(pars)
    check_tune_algo(pars)

    if pars["save_estimators"]:
        os.makedirs(pars["output_dir"] / "estimators", exist_ok=True)

    if pars["estimator"] == "finetunetabpfn":
        folder_stats_finetune = pars["output_dir"] / "stats_finetune"
        os.makedirs(folder_stats_finetune, exist_ok=True)

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

    dict_results = defaultdict(list)
    if pars["tune"]: dict_hpo = defaultdict(list)

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

        fix_estimator_fixed_params_during_resampling_(
            estimator=estimator,
            repeat=repetition,
            fold=fold,
            resample_program_params=pars
        )

        t = time()
        estimator.fit(X_train, y_train)
        fit_time = time() - t
        
        fit_preprocessing_dict: dict = estimator.collect_fit_preprocessing_info()

        if pars["tune"]:
            best_hps = estimator.get_best_hps()
            refit_time = estimator.get_refit_time()
            search_losses = estimator.get_search_losses()
            best_loss = np.nanmin(search_losses)
            search_losses_dict = {f"loss_{i}": value_loss for i, value_loss in enumerate(search_losses)}
    
        logger.debug("\t-Estimator fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")

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
                refit_time=refit_time,
                **best_hps,
                best_loss=best_loss,
                **search_losses_dict
            )
        
        if pars["save_estimators"]:
            estimator.save(get_estimator_filepath(pars, repetition, fold))
     
        if pars["estimator"] == "finetunetabpfn":
            iteration_signature = get_resample_iteration_signature(repetition, fold)
            estimator.save_finetune_stats(
                txt_filepath = folder_stats_finetune / f"df_finetune_{iteration_signature}.txt",
                json_filepath = folder_stats_finetune / f"stats_finetune_{iteration_signature}.json"
            )

        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")
    

    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    hpo_filepath = output_dir / "hpo.txt"
    configuration_filepath = output_dir / "configuration.json"
    
    df_pred_results = PredictionDataframe()
    
    tune_hps_configuration = None \
        if pars["tune_configuration"] is None \
        else pars["tune_configuration"]["configuration"]

    df_pred_results.build_from_data(
        **dict_results, 
        save_path=output_dir,
        estimator=pars["estimator"],
        predict_dataset=name_dataset,
        splitting_mode=pars["splitting_mode"],
        preprocessing=pars["preprocessing"],
        tune=pars["tune"],
        tune_hps_configuration=tune_hps_configuration,
        n_threads=pars["nthreads"]
    )

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)


    if pars["tune"]:
        # remove params_distributions from the tune conf to store it in a df
        del pars["tune_configuration"]["params_distributions"]
        dict_hpo = {
            "dataset": name_dataset,
            "estimator": pars["estimator"],
            "splitting_mode": pars["splitting_mode"], 
            "preprocessing": pars["preprocessing"],
            **pars["tune_configuration"],
            **dict_hpo
        }
        pd.DataFrame(dict_hpo).to_csv(hpo_filepath, sep="\t", index=False)

    create_json_configuration_file(pars, configuration_filepath)
    logger.debug(f"Outputs created at {output_dir}")




if __name__ == "__main__":
    main()