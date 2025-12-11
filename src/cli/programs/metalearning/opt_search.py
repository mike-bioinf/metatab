"""
Program to run an optmization search on some data with some ml algo.
The program allows to specify the search settings controlled by the tune arguments parser.
"""

from __future__ import annotations

import sys
import argparse
from time import time
from typing import TYPE_CHECKING
from metatab_utils.data_loader import DataLoader
from hp_search.config import ConfigSearchCV
from estimators.utils.pick import pick_estimator_class
from cli.parser import make_base_parser, make_tune_parser

from cli.helper import (
    check_target_feature,
    adjust_io_paths_,
    manage_output_path,
    resolve_preprocessing_info,
    create_logger,
    build_early_stop_configuration,
    build_tune_configuration
)

if TYPE_CHECKING:
    from logging import Logger
    from estimators.estimators import Estimator




def parse_args(args):
    p = argparse.ArgumentParser(parents=[make_base_parser(), make_tune_parser()])
    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    p.add_argument("-o", "--output-file", required=True, help="Output filepath.")
    p.add_argument("--seed", default=42, type=int, help="Seed used to control randomness.")
    p.add_argument("--save-realtime", action="store_true", 
                help="""Enables to save the search results after every search iteration. 
                Adds a bit of overhead. Highly suggested for long jobs.""")
    return p.parse_args(args)



def log_program_setting(logger: Logger, pars: dict, name_dataset: str):
    logger.debug((
        f"\nLaunching {pars["tune_algo"]} search on {name_dataset} with"
        f" {pars["estimator"]} on the {pars["tune_space"]} tune space, with"
        f" {pars["tune_n_iter"]} iterations and {pars["tune_n_cv_repeats"]}-repeat {pars["tune_n_cv_folds"]}-fold cv.\n"
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
    check_n_iter(pars)
    
    adjust_io_paths_(pars, "input_data", "output_file")
    manage_output_path(pars, "output_file", False)
    
    early_stop_configuration = build_early_stop_configuration(pars)
    tune_configuration = build_tune_configuration(pars)

    # set instruction for building and saving the search data
    ConfigSearchCV.refit_with_best_hps = False
    ConfigSearchCV.build_df_search = True
    # if pars["save_realtime"]:
    #     ConfigSearchCV.save_realtime_df_search_filepath = pars["output_file"]
    
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y, name_dataset = dl.X, dl.y, dl.generic_dataset_name
    
    log_program_setting(logger, pars, name_dataset)
    logger.debug("Data loaded in memory!")
    
    estimator_class = pick_estimator_class(pars["estimator"], "tune")

    estimator: Estimator = estimator_class(
        # here we resolve the preprocessing info in order to store
        # the explicit preprocessing in the search data
        preprocessing=resolve_preprocessing_info(pars),
        seed=pars["seed"],
        n_threads=pars["nthreads"],
        early_stop_configuration=early_stop_configuration,
        tune_configuration=tune_configuration
    )

    starting_time = time()
    estimator.fit(X, y)
    fit_time = round(((time() - starting_time)/60), ndigits=2)
    logger.debug(f"Completed search with runtime of {fit_time} minutes.")

    if not pars["save_realtime"]:
        df_search = estimator.estimator_.df_search_
        df_search.to_csv(pars["output_file"], sep="\t", index=False)

    logger.debug(f"File created at location: {pars['output_file']}")




if __name__ == "__main__":
    main()