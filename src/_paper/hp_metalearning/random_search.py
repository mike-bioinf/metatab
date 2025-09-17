from __future__ import annotations

import sys
import argparse
from typing import TYPE_CHECKING
from estimators import Estimator
from metatab_utils.general import create_logger
from metatab_utils.data_loader import DataLoader

from metatab_utils.helper_programs import (
    check_target_feature,
    adjust_io_paths_,
    manage_output_path,
    adjust_early_stopping_rounds_,
    _pick_params_distributions_configuration,
    pick_estimator_class,
)

if TYPE_CHECKING:
    from logging import Logger




def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    
    p.add_argument("-o", "--output-file", required=True, help="Output filepath.")
    
    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"], 
                   help="Defines the data input format. One between 'xy' and 'df'.")
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")
    
    p.add_argument("-e", "--estimator", required=True, choices=["random_forest", "xgb", "es_xgb", "catboost", "es_catboost", "lgbm", "es_lgbm", "tabpfn"],
                   help="A tunable estimator.")
    
    p.add_argument("-p", "--preprocessing", default="base", choices=["base", "density_filter", "pca"])
    
    p.add_argument("-r", "--early-stopping-rounds", type=int, default=-1,
                   help="""Number of early stop rounds to use when using the "es" estimators.
                   The default is -1, which means 100 for the "es" estimators and a value to be
                   ignored by the "non es" estimators (in this case other values will results in an error).""")
    
    p.add_argument("--tune-space", default="default", help="Tune space.")

    p.add_argument("--niter", default=1500, type=int, help="Number of search iterations.")
    
    p.add_argument("--nrepeats", default=3, type=int, help="Number of inner cv repeats.")

    p.add_argument("--nfolds", default=5, type=int, help="Numbe of inner cv folds.")
    
    p.add_argument("--seed", default=42, type=int, help="Seed used to control randomness.")

    p.add_argument("--nthreads", default=16, type=int, help="Number of CPU threads to use. Defaults to 16.")

    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")



def log_program_setting(logger: Logger, pars: dict, name_dataset: str):
    logger.debug(
        f"\nLaunching random search on {name_dataset} with \
        {pars["niter"]} iterations and {pars["nrepeats"]}-repeat {pars["nfolds"]}-fold cv."
    )





def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_target_feature(pars)
    adjust_io_paths_(pars, "input_data", "output_file")
    manage_output_path(pars, "output_file", False)
    adjust_early_stopping_rounds_(pars)
    pars["tune"] = True
    
    # we build the dict needed by the Estimator
    pars["tune_configuration"] = {
        "configuration": pars["tune_space"],
        "algo": "random",
        "n_repeats": pars["nrepeats"],
        "n_splits": pars["nfolds"]
    }

    # add the hps space
    pars["tune_configuration"]["params_distributions"] = (
        _pick_params_distributions_configuration(pars)
    )

    logger = create_logger(sys.stdout)
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y, name_dataset = dl.X, dl.y, dl.generic_dataset_name
    
    log_program_setting(logger, pars, name_dataset)
    logger.debug("Data loaded in memory!\n")
    
    estimator_class = pick_estimator_class(pars)

    estimator: Estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"],
        n_threads=pars["nthreads"],
        early_stopping_rounds=pars["early_stopping_rounds"],
        tune_configuration=pars["tune_configuration"]
    )


    ### TODO: modify search cv in order to take train info, and select the metric ??? (AUC, logloss)

    ### TODO: modify searchcv in order to build the hpo.txt. 
    # it has to build the df internally and then in resample program we take it from it
    # even though we have to accumulate in the resample loop and then concat
    
    ### TODO: z-normalize target metric

    estimator.fit(X, y)
    df_search = estimator.estimator_.df_search_

    ## TODO aggregate cv results for single confs (1 row --> 1 conf-value)
    ## TODO: add preprocessing column, add metafeatures (must be obtained here in the main program)

    ## TODO: save
    df_search.to_csv()





if __name__ == "main":
    main()