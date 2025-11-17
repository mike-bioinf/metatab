"""
Program to fit an ML estimator on a dataset.
The program seralizes the fitted model in a binary file via pickle.
"""

import sys
import argparse
from estimators.estimators import Estimator
from metatab_utils.data_loader import DataLoader

from cli.helper import (
    adjust_io_paths_, 
    manage_output_path,
    check_fit_resample_args,
    pick_estimator_class,
    check_y_is_integer_encoded,
    create_logger,
    build_early_stop_configuration,
    build_tune_configuration,
    resolve_preprocessing_info
)

from cli.parser import (
    make_base_parser,
    make_extra_fit_parser,
    make_tune_parser
)



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sub_estimator_mode = p.add_subparsers(required=True, title="Estimator Mode", description="valid subcommands")
    p_default = sub_estimator_mode.add_parser("default", parents=[make_base_parser(), make_extra_fit_parser()])
    p_tune = sub_estimator_mode.add_parser("tune", parents=[make_base_parser(), make_extra_fit_parser(), make_tune_parser()])
    p_default.set_defaults(estimator_mode="default")
    p_tune.set_defaults(estimator_mode="tune")
    return p.parse_args(args)



def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))
    check_fit_resample_args(pars, logger)

    adjust_io_paths_(pars, "input_data", "output_path")
    manage_output_path(pars, "output_path", False)

    early_stop_configuration = build_early_stop_configuration(pars)
    tune_configuration = build_tune_configuration(pars)

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
        preprocessing=resolve_preprocessing_info(pars),
        seed=pars["seed"],
        n_threads=pars["nthreads"],
        early_stop_configuration=early_stop_configuration,
        tune_configuration=tune_configuration
    )
    
    # if pars["estimator"] == "autotabpfn":
    #     out_file = pars["output_path"]
    #     # here we use the basename of the out file to have unique directory names
    #     models_dir = out_file.parent / f"autogluon_fitted_models_{out_file.stem}"
    #     estimator.set_directory_save_models(models_dir, create_dir=True)

    estimator.fit(X_train, y_train)
    logger.debug("Estimator fitted on training data.")
    
    # we set y_train and fit_dataset_name since are requested by the predict program 
    estimator._y_train_ = y_train
    estimator._fit_dataset_name_ = fit_dataset_name
    
    # if pars["estimator"] == "finetunetabpfn":
    #     out_file = pars["output_path"]
    #     estimator.save_finetune_stats(
    #         txt_filepath=out_file.parent / f"df_finetune_{out_file.stem}.txt",
    #         json_filepath=out_file.parent / f"stats_finetune_{out_file.stem}.txt"
    #     )
    
    estimator.save(pars["output_path"], check_is_fitted=True)
    logger.debug(f"Estimator serialized in '{pars["output_path"]}'.")




if __name__ == "__main__":
    main()