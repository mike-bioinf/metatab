"""
Program to run an optimization search on some data with some ml algo.
The program allows to specify the search settings controlled by the tune parser.
"""

from __future__ import annotations

import sys
import argparse
from time import time
from typing import TYPE_CHECKING
from metatab.metatab_utils.data_loader import DataLoader
from metatab.hp_search.config import ConfigSearchCV
from metatab.estimators.utils.pick import pick_estimator_class
from metatab.estimators.utils.general import check_y_is_integer_encoded

from metatab.cli.parser import (
    make_base_parser, 
    make_tune_parser, 
    make_extra_base_parser
)

from metatab.cli.helper import (
    check_target_feature,
    check_early_stop_parameters,
    adjust_io_paths_,
    manage_output_path,
    create_logger,
    build_early_stop_configuration,
    build_tune_configuration
)

if TYPE_CHECKING:
    from logging import Logger
    from metatab.estimators.estimators import Estimator




def parse_args(args):
    p = argparse.ArgumentParser(parents=[make_base_parser(), make_extra_base_parser(), make_tune_parser()])
    p.add_argument("--output-file", required=True, help="Name of the file created in output at '--output-dir' location.")
    p.add_argument("--seed", default=42, type=int, help="Seed used to control randomness.")
    return p.parse_args(args)



def log_program_setting(logger: Logger, pars: dict, name_dataset: str):
    logger.debug((
        f"\nLaunching {pars["tune_algo"]} search on {name_dataset} with"
        f" {pars["estimator"]} on the {pars["tune_space"]} tune space, with"
        f" {pars["tune_n_iter"]} iterations and {pars["tune_n_cv_repeats"]}-repeat {pars["tune_n_cv_folds"]}-fold cv."
    ))



def check_n_iter(pars: dict) -> None:
    '''
    SearchCV skips the evaluation when optimizing for a single point.
    The df search is not constructed in this scenario.
    '''
    if pars["tune_n_iter"] == 1:
        raise ValueError(
            "Is not possible to collect the search data when 'tune_n_iter' equal 1."
        )



def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))

    check_target_feature(pars)
    check_early_stop_parameters(pars)
    check_n_iter(pars)
    
    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    
    early_stop_configuration = build_early_stop_configuration(pars)
    tune_configuration = build_tune_configuration(pars)

    # set instruction for building the search data
    ConfigSearchCV.refit_with_best_hps = False
    ConfigSearchCV.build_df_search = True
    
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y, name_dataset = dl.X, dl.y, dl.generic_dataset_name
    check_y_is_integer_encoded(y)

    log_program_setting(logger, pars, name_dataset)
    logger.debug("Data loaded in memory!")
    
    estimator_class = pick_estimator_class(pars["estimator"], "tune")

    estimator: Estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"],
        n_threads=pars["nthreads"],
        device=pars["device"],
        early_stop_configuration=early_stop_configuration,
        tune_configuration=tune_configuration
    )

    starting_time = time()
    estimator.fit(X, y)
    fit_time = round(((time() - starting_time)/60), ndigits=2)
    logger.debug(f"Completed search with runtime of {fit_time} minutes.")

    output_filepath = pars["output_dir"] / pars["output_file"]
    df_search = estimator.estimator_.df_search_
    df_search.to_csv(output_filepath, sep="\t", index=False)
    logger.debug(f"Output file created at: {output_filepath}.")




if __name__ == "__main__":
    main()