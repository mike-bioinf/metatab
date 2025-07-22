import argparse
from typing import Any
from ast import literal_eval



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # positional arguments
    p.add_argument("-i", "--input-data",  required=True, help="Path to the dataset folder/file.")
    
    p.add_argument("-o", "--output-dir", required=True, default=".",
                   help="""Path of the folder storing the results.
                   If not provided the folder in which the program is run is used.""")
    
    p.add_argument("-e", "--estimator", required=True, choices=["random_forest", "xgb", "es_xgb", "tabpfn"], 
                    help="""ML 'estimator' to use. One of 'random_forest', 'xgb', 'es_xgb', 'tabpfn'.""")

    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"],
                    help="Defines the data input format. One between 'xy' and 'df'.")
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")

    p.add_argument("-n", "--splitting-mode", default="cv", choices=["holdout", "cv"],
                    help="Defines the splitting strategy. One between 'holdout' and 'cv'(default).")

    p.add_argument("-d", "--splitting-specs", default=None, 
                   help="""Specifies splitting details as key=value pairs. 
                    Defaults to {'n_repeats':10, 'n_splits':5} with 'cv' --splitting-mode.
                    Defaults to {'n_splits':50, 'train_size':0.9} with 'holdout' --splitting-mode.""")

    p.add_argument("-p", "--preprocessing", default="base", choices=["base", "density_filter", "pca"],
                    help= """Preprocessing to apply on the feature space. One of 'base', 'density_filter' and 'pca'.
                    -base: a general minimal preprocessing is applied according to the used estimator.
                    -density_filter: The number of columns is reduced to 500 keeping only the most dense features.
                    Note: according to the type of filtering applied the exact number of selected features may be not exactly 500. 
                    This strategy is automatically selected based on the estimator used (no user control over it).
                    -pca: PCA is applied and only the first N principal components retaining the 95 percent of the variance are kept.""")

    p.add_argument("-s", "--seed", default=42, type=int, 
                    help="""Seed used to control randomness.
                    In particular it controls the randomness inherent to the splitting procedures.
                    Important note: this seed does NOT control the randomness of the models.""")

    p.add_argument("-t", "--tune", action="store_true", 
                   help="""Tune the estimator hyperparameters. The tuning strategy as well as the HPs to tune and
                   the tested values/strategies are not customizable. They are picked according to the estimator used.
                   Not all estimators can be tuned. For tabpfn a separated estimator must be used for tuning.
                   In these cases setting this parameter will result in an error.""")
    
    ## TODO: to remove in production once find good defaults
    p.add_argument("-c", "--hps-configuration", default=None, 
                   help="""Allow to specify which configurations of HPS to use for tuning.
                   If None, the default, is specified and the estimator is tunable 
                   and the --tune parameter is set, then the default hps configuration is used.
                   These configurations are indicated following the schema 'c+number' (i.e, c1).""")
    
    p.add_argument("-q", "--save-estimators", action="store_true",
                   help="""Option to save the fitted estimators. 
                   The estimators object are saved via pickle in the 'estimators' folder.
                   Note that all estimators fitted during the splitting procedure are saved.
                   The filenames follow the generic structure: {estimator}__{preprocessing}__{repetition}{fold}.
                   In case of 'holdout' splitting mode the "__{repetition}{fold}" part is replaced by
                   a sequential number.""")

    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p.parse_args(args)




def adjust_splitting_specs_(pars: dict) -> None:
    '''
    Adjustment of splitting_specs parameter.
    Modifies the dictionary of arguments in place.
    '''
    splitting_mode = pars["splitting_mode"]
    splitting_specs = pars["splitting_specs"]
    
    if splitting_specs is not None:
        splitting_specs = _try_parse_specs_into_dict(pars["splitting_specs"], "--splitting-specs")
        _check_splitting_specs_keys(splitting_mode, splitting_specs)
    
    if splitting_mode == "cv" and splitting_specs is None:
        splitting_specs = {"n_repeats": 10, "n_splits": 5}

    if splitting_mode == "holdout" and splitting_specs is None:
        splitting_specs = {"n_splits": 50, "train_size": 0.9}

    pars["splitting_specs"] = splitting_specs



def _check_splitting_specs_keys(splitting_mode: str, splitting_specs: dict) -> None:
    '''
    Checks the presence of mandatory splitting specifications (keys) 
    based on the splitting mode.
    '''
    expected_specs = ["n_splits", "train_size"] if splitting_mode == "holdout" else ["n_repeats", "n_splits"]
    for spec in expected_specs:
        if spec not in splitting_specs.keys():
            raise ValueError(
                f"With '{splitting_mode}' --splitting-mode you must pass the {expected_specs} keys in --splitting-specs."
            )
        


def _try_parse_specs_into_dict(specs: str, error_message_specs: str) -> dict[str, Any]:
    '''Utility to parse the string dict representation to a dict'''
    try:
        specs = literal_eval(specs)
    except Exception:
        raise ValueError(
            f"{error_message_specs} " + "cannot be correctly parsed into a dict. \
            It should be passed following the syntax '{'key': value, ...}'.\
            Remember to enclose the keys in ticks ('') if they are python strings."
        )
    return specs
