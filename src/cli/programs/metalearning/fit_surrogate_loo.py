"""Program to fit and save surrogate frameworks on meta-data in a leave-one-out fashion.

The meta-data is expected to be contained in a folder in multiple files.
The files are expected to be tab-separated txt files.
The program loop over these files excluding one at a time and fitting 
the surrogate model on the others.
"""
from __future__ import annotations

import sys
import argparse
import warnings
import joblib
import pandas as pd
from copy import deepcopy
from typing import TYPE_CHECKING
from sklearn.pipeline import make_pipeline
from metalearning.surrogate_rf import SurrogateRandomForestRegressor
from metalearning.encode.encode import get_encoding_scheme

from cli.helper import (
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
    
    p.add_argument("-o", "--output-folder", required=True, 
                   help="""Output folder path. It will contains all the surrogate models called after the names
                   of the dataset excluded with a joblib extension.""")

    p.add_argument("-y", "--column-metric", default="z_normalized_loss", 
                   help="Name of the perfomance metric column to treat as the y target.")

    p.add_argument("-e", "--estimator", required=True, 
                   choices=["random_forest", "xgb", "es_xgb", "catboost", "es_catboost", "lgbm", "es_lgbm", "tabpfn"],
                   help="""The estimator on which the meta-folder has been generated. 
                   Needed to apply the correct preprocessing to the meta-data.""")
    
    p.add_argument("--seed", default=42, type=int, help="Seed used to control randomness of the surrogate model.")

    p.add_argument("--nthreads", default=16, type=int, help="Number of CPU threads to use. Defaults to 16.")

    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p.parse_args(args)



def log_program_setting(logger: Logger, pars: dict):
    logger.debug(
        (
            f"\nLaunching the surrogate fit LOO program on folder '{pars["meta_folder"].name}'"
            f" with '{pars["estimator"]}' preprocessing scheme.\n"
        )
    )



def load_datasets(folder: Path) -> dict[str, pd.DataFrame]:
    '''
    Load the dataset and insert them in a dict with keys equal to the basename of the files.
    '''
    dfs = {}
    for file in folder.iterdir():
        if not file.is_file():
            raise ValueError(f"Unexpected folder in metafolder: {file.name}")
        try:
            basename = file.stem
            dfs[basename] = pd.read_csv(file, sep="\t")
        except Exception as e:
            raise ValueError(
                f"The following error is encountered when reading the file '{file.name}': {e}"
            )
    return dfs




def main():
    pars = vars(parse_args(sys.argv[1:]))
    adjust_io_paths_(pars, "meta_folder", "output_folder")
    manage_output_path(pars, "output_folder", True)

    logger = create_logger(sys.stdout)
    log_program_setting(logger, pars)

    datasets = load_datasets(pars["meta_folder"])
    logger.debug("Meta-data loaded in memory!\n")
    
    y_col = pars["column_metric"]
    preprocessor = get_encoding_scheme(pars["estimator"])
    surrogate_rf = SurrogateRandomForestRegressor(n_jobs=pars["nthreads"], random_state=pars["seed"])
    surrogate_framework = make_pipeline(*preprocessor, surrogate_rf).set_output(transform="pandas")
    
    # check on y column for all datasets
    for name_dataset, dataset in datasets.items():
        if y_col not in dataset.columns:
            raise ValueError(f"'{y_col}' column_metric not found in '{name_dataset}'.")
    
        if dataset[y_col].isna().any():
            raise ValueError(f"NA in column_metric in '{name_dataset}'.")

    # leave one out fitting
    for name_dataset, dataset in datasets.items():
        # the copy is for safety even though it should work also 
        # refitting the original pipeline multiple times
        copy_surrogate_framework = deepcopy(surrogate_framework)
        name_output_file = pars["output_folder"] / f"{name_dataset}.joblib"
        
        meta_data = pd.concat(
            [d for n, d in datasets.items() if n != name_dataset],
            axis=0,
            ignore_index=True
        )

        X = meta_data.drop(columns=y_col)
        y = meta_data[y_col]
        
        # suppress variance threshold warnings on full na slices
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="Degrees of freedom <= 0 for slice.*", category=RuntimeWarning)
            warnings.filterwarnings(action="ignore",message="All-NaN slice encountered", category=RuntimeWarning)
            copy_surrogate_framework.fit(X, y)

        logger.debug(f"Surrogate framework fitted excluding '{name_dataset}'.")

        with open(name_output_file, "wb") as f:
            joblib.dump(copy_surrogate_framework, f)

        logger.debug(f"Surrogate framework serialized in: {name_output_file}\n")




if __name__ == "__main__":
    main()