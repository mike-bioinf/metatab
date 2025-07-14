import os
import argparse
from pathlib import Path



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-i", "--input-path", required=True, help="Path to the dataset folder/file.")
    p.add_argument("-o", "--output-path", required=True, help="Path of the pickle file created in output.")

    p.add_argument("-m", "--input-mode", required=True, choices=["sets", "xy", "df"],
                    help="Define the expected form of the input. One of 'sets', 'xy', or 'df'.")

    p.add_argument("-e", "--estimator", required=True, 
                    choices=["random_forest", "xgb", "es_xgb", "tabpfn"], 
                    help=""" ML 'estimator' to use. One of 'random_forest', 'xgb', 'es_xgb', 'tabpfn'.""")
    
    p.add_argument("-p", "--preprocessing", default="base", choices=["base", "density_filter", "pca"],
                    help= """Preprocessing to apply on the feature space. One of 'base', 'density_filter' and 'pca'.
                    -base: a general minimal preprocessing is applied according to the used estimator.
                    -density_filter: The number of columns is reduced to 500 keeping only the most dense features.
                    Note: according to the type of filtering applied the exact number of filtered features may be not exactly 500. 
                    This strategy is automatically selected based on the estimator used (no user control over it).
                    -pca: PCA is applied and only the first N principal components retaining the 95 percent of the variance are kept.""")
    
    p.add_argument("-t", "--tune", action="store_true", 
                   help="""Tune the estimator hyperparameters. The tuning strategy as well as the HPs to tune and
                   the tested values/strategies are not customizable. They are picked according to the estimator used.
                   Not all estimators can be tuned. For tabpfn a separated estimator must be used for tuning.
                   In these cases setting this parameter will result in an error.""")

    p.add_argument("-y", "--target-feature", default=None, 
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")

    p.add_argument("-s", "--seed", default=42, type=int, help="Seed used to control randomness.")

    p.add_argument("--create-outdir", action="store_true",
                   help="Create the output folder if does not exists.")

    return p.parse_args(args)




def manage_output_path(pars: dict) -> None:
    '''
    Control whether the output folder exists and whether to create it if not.
    Assumes that output path is a Path object.
    '''
    out: Path = pars["output_path"]
    out_folder = out.parents[0]

    if not out_folder.exists() and not pars["create_outdir"]:
        raise FileNotFoundError(f"{out_folder} does not exists!")
    elif not out_folder.exists() and pars["create_outdir"]:
        os.makedirs(out_folder)


def check_args(pars: dict) -> None:
    '''General check on arguments'''
    check_target_feature(pars)
    check_not_tunable_estimators(pars)


def check_target_feature(pars: dict) -> None:
    '''Check that the target feature is set with df input-mode'''
    if pars["input_mode"] == "df" and pars["target_feature"] is None:
        raise ValueError("--target-feature must be specified with 'df' input mode.")


## TODO: complete with the tune-tabpfn estimator name
def check_not_tunable_estimators(pars: dict) -> None:
    '''Check whether the tune flag is used with not tunable estimator'''
    if pars["estimator"] == "tabpfn":
        raise ValueError(
            "The 'tabpfn' estimator cannot be tuned setting --tune. Use the '' estimator."
        )
