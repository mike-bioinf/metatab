"""
Program to fit an ML estimator on a dataset with different preprocessing and tuning options.
The program saves the fitted and serialized model in a binary file via pickle.
"""

import sys
from estimators import Estimator
from metatab_utils.data_loader import DataLoader
from cli.fit.params import parse_args

from metatab_utils.helper_programs import (
    adjust_io_paths_, 
    manage_output_path,
    check_fit_resample_args,
    check_tune_configuration,
    adjust_tune_configuration_arg_,
    adjust_early_stopping_rounds_,
    pick_estimator_class,
    check_y_is_integer_encoded,
    create_logger, 
)




def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))
    check_fit_resample_args(pars)

    adjust_io_paths_(pars, "input_data", "output_path")
    manage_output_path(pars, "output_path", False)
    adjust_tune_configuration_arg_(pars)
    adjust_early_stopping_rounds_(pars)
    check_tune_configuration(pars, logger)
    
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="train",
        skip=["X_test", "y_test"]
    )

    X_train, y_train = dl.X_train, dl.y_train
    check_y_is_integer_encoded(y_train)
    logger.debug("Data loaded in memory!")

    # here we consider all 3 load methods to retrieve the dataset_name
    fit_dataset_name = dl.train_dataset_name if dl.train_dataset_name else dl.generic_dataset_name
    estimator_class = pick_estimator_class(pars)
    
    estimator: Estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"],
        n_threads=pars["nthreads"],
        early_stopping_rounds=pars["early_stopping_rounds"],
        tune_configuration=pars["tune_configuration"]
    )
    
    if pars["estimator"] == "autotabpfn":
        out_file = pars["output_path"]
        # here we use the basename of the out file to have unique directory names
        models_dir = out_file.parent / f"autogluon_fitted_models_{out_file.stem}"
        estimator.set_directory_save_models(models_dir, create_dir=True)

    estimator.fit(X_train, y_train)
    logger.debug("Estimator fitted on training data.")
    
    # we set y_train and fit_dataset_name since are requested by the predict program 
    estimator._y_train_ = y_train
    estimator._fit_dataset_name_ = fit_dataset_name
    
    if pars["estimator"] == "finetunetabpfn":
        out_file = pars["output_path"]
        estimator.save_finetune_stats(
            txt_filepath=out_file.parent / f"df_finetune_{out_file.stem}.txt",
            json_filepath=out_file.parent / f"stats_finetune_{out_file.stem}.txt"
        )
    
    estimator.save(pars["output_path"], check_is_fitted=True)
    logger.debug(f"Estimator serialized in '{pars["output_path"]}'.")




if __name__ == "__main__":
    main()