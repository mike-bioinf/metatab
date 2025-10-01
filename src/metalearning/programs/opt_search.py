"""Program to run an optmization search on some data with some ml algo.

The program allows to specify the search settings like optimization algorithm, 
tune space, number of iterations and the specifics of the inner cross validation.
"""

from __future__ import annotations

import sys
import argparse
from time import time
from typing import TYPE_CHECKING
from estimators import Estimator
from metatab_utils.data_loader import DataLoader
from hp_search.utils import ConfigSearchCV

from metatab_utils.helper_programs import (
    check_target_feature,
    adjust_io_paths_,
    manage_output_path,
    adjust_early_stopping_rounds_,
    _pick_params_distributions_configuration,
    pick_estimator_class,
    create_logger
)

if TYPE_CHECKING:
    from logging import Logger




def parse_args(args):
    p = argparse.ArgumentParser()
    
    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    
    p.add_argument("-o", "--output-file", required=True, help="Output filepath.")
    
    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"], 
                   help="Defines the data input format. One between 'xy' and 'df'.")
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")
    
    p.add_argument("-e", "--estimator", required=True, 
                   choices=["random_forest", "xgb", "es_xgb", "catboost", "es_catboost", "lgbm", "es_lgbm", "tabpfn"],
                   help="The estimator.")
    
    p.add_argument("-p", "--preprocessing", default="base", choices=["base", "density_filter", "pca"])
    
    p.add_argument("-r", "--early-stopping-rounds", type=int, default=-1,
                   help="""Number of early stop rounds to use when using the "es" estimators.
                   The default is -1, which means 100 for the "es" estimators and a value to be
                   ignored by the "non es" estimators (in this case other values will results in an error).""")
    
    p.add_argument("--tune-algo", default="random", choices=["random", "tpe"], help="Tune optimization algorithm.")

    p.add_argument("--tune-space", default="default", help="Tune space")

    p.add_argument("--niter", default=1500, type=int, help="Number of search iterations. Cannot be 1.")
    
    p.add_argument("--nrepeats", default=3, type=int, help="Number of inner cv repeats.")

    p.add_argument("--nfolds", default=5, type=int, help="Numbe of inner cv folds.")
    
    p.add_argument("--seed", default=42, type=int, help="Seed used to control randomness.")

    p.add_argument("--nthreads", default=16, type=int, help="Number of CPU threads to use. Defaults to 16.")

    p.add_argument("--save-realtime", action="store_true", 
                   help="""Enables to save the search results after every search iteration.
                   Adds a bit of overhead. Highly suggested for long jobs.""")

    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p.parse_args(args)



def log_program_setting(logger: Logger, pars: dict, name_dataset: str):
    logger.debug(
        (
            f"\nLaunching {pars["tune_algo"]} search on {name_dataset} with"
            f" {pars["estimator"]} on the {pars["tune_space"]} tune space, with"
            f" {pars["niter"]} iterations and {pars["nrepeats"]}-repeat {pars["nfolds"]}-fold cv.\n"
        )
    )



def check_n_iter(pars: dict) -> None:
    '''
    SearchCV skips the evaluation when optimizing for a single point.
    The df search is not constructed in this scenario.
    '''
    if pars["niter"] == 1:
        raise ValueError(
            "Is not possible to collect the search data when 'niter' equal 1."
        )




def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))
    check_target_feature(pars)
    check_n_iter(pars)
    adjust_io_paths_(pars, "input_data", "output_file")
    manage_output_path(pars, "output_file", False)
    adjust_early_stopping_rounds_(pars)
    pars["tune"] = True
    
    # set instruction for building and saving the search data
    ConfigSearchCV.refit_with_best_hps = False
    ConfigSearchCV.build_df_search = True
    if pars["save_realtime"]:
        ConfigSearchCV.save_realtime_df_search_filepath = pars["output_file"]
    
    # build the tune dict needed by the Estimator
    pars["tune_configuration"] = {
        "configuration": pars["tune_space"],
        "algo": pars["tune_algo"],
        "n_iter": pars["niter"],
        "n_repeats": pars["nrepeats"],
        "n_splits": pars["nfolds"]
    }

    # add the hps space
    pars["tune_configuration"]["params_distributions"] = (
        _pick_params_distributions_configuration(pars)
    )

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
    
    estimator_class = pick_estimator_class(pars)

    estimator: Estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"],
        n_threads=pars["nthreads"],
        early_stopping_rounds=pars["early_stopping_rounds"],
        tune_configuration=pars["tune_configuration"]
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