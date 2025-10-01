"""Program to fit and save surrogate frameworks on meta-data.

The meta-data is expected to be contained in a folder in multiple files.
The files are expected to be tab-separated txt files.
"""
from __future__ import annotations

import sys
import argparse
import joblib
import pandas as pd
from typing import TYPE_CHECKING
from sklearn.pipeline import make_pipeline
from metalearning.surrogate_rf import SurrogateRandomForestRegressor
from metalearning.encoding import get_encoding_scheme

from metatab_utils.helper_programs import (
    adjust_io_paths_,
    manage_output_path,
    create_logger
)

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path




def parse_args(args):
    p = argparse.ArgumentParser()
    
    p.add_argument("-i", "--meta-folder", required=True, help="Path of the folder containing the meta-datasets.")
    
    p.add_argument("-o", "--output-file", required=True, 
                   help="Output filepath. It will be a binary file containing the serialized surrogate model.")

    p.add_argument("-y", "--column-metric", default="z_normalized_loss", 
                   help="Name of the perfomance metric column tratead as the y target.")

    p.add_argument("-e", "--estimator", required=True, 
                   choices=["random_forest", "xgb", "catboost", "lgbm", "tabpfn"],
                   help="""The estimator on which the meta-datasets have been generated. 
                   Needed to apply the correct preprocessing to the meta-data.
                   The early stopped estimator version are not present since they require 
                   the same preprocessing of their base counterpart.""")
    
    p.add_argument("--seed", default=42, type=int, help="Seed used to control randomness of the surrogate model.")

    p.add_argument("--nthreads", default=16, type=int, help="Number of CPU threads to use. Defaults to 16.")

    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p.parse_args(args)



def log_program_setting(logger: Logger, pars: dict):
    logger.debug(
        (
            f"\nLaunching the surrogate fit program on folder '{pars["meta_folder"].stem}'"
            f" with '{pars["estimator"]}' preprocessing scheme.\n"
        )
    )



def load_datasets(folder: Path) -> list[pd.DataFrame]:
    dfs = []
    for file in folder.iterdir():
        if not file.is_file():
            raise ValueError(f"Unexpected folder in metafolder: {file.stem}")
        try:
            dfs.append(pd.read_csv(file, sep="\t"))
        except Exception as e:
            raise ValueError(
                f"The following error is encountered when loading the file '{file.stem}': {e}"
            )
    return dfs




def main():
    pars = vars(parse_args(sys.argv[1:]))
    adjust_io_paths_(pars, "meta_folder", "output_file")
    manage_output_path(pars, "output_file", False)

    logger = create_logger(sys.stdout)
    log_program_setting(logger, pars)

    datasets = load_datasets(pars["meta_folder"])
    meta_data = pd.concat(datasets, axis=0, ignore_index=True)
    logger.debug("Meta-data loaded in memory!")

    y_col = pars["column_metric"]
    if y_col not in meta_data.columns:
        raise ValueError(f"'{y_col}'column_metric not found in meta_data.")
    
    preprocessor = get_encoding_scheme(pars["estimator"])
    surrogate_rf = SurrogateRandomForestRegressor(n_jobs=pars["nthreads"], random_state=pars["seed"])
    surrogate_framework = make_pipeline(*preprocessor, surrogate_rf)
   
    X = meta_data.drop(columns=y_col)
    y = meta_data[y_col]

    surrogate_framework.fit(X, y)
    logger.debug("Surrogate framework fitted on meta-data.")

    with open(pars["output_file"], "wb") as f:
        joblib.dump(surrogate_framework, f)

    logger.debug(f"Surrogate framework serialized in: {pars["output_file"]}")




if __name__ == "__main__":
    main()