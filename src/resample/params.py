import argparse
from metatab_utils.helper_params import try_parse_specs_into_dict



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-i", "--input-data",  required=True, help="Path to the dataset folder/file.")
    
    p.add_argument("-o", "--output-dir", required=True, default=".",
                   help="""Path of the folder storing the results.
                   If not provided the folder in which the program is run is used.""")
    
    p.add_argument("-e", "--estimator", required=True, choices=["random_forest", "xgb", "es_xgb", "catboost", "es_catboost",
                    "tabpfn"], 
                    help="""ML estimator to use. One of 'random_forest', 'xgb', 'es_xgb', 'catboost', 'es_catboost', 
                    'tabpfn'.""")

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
                    In particular it controls the randomness inherent to the estimators, splitting and tuning procedures.""")

    p.add_argument("-t", "--tune", action="store_true", 
                   help="""Tune the estimator hyperparameters. 
                   We use random search as tuning strategy and a preset of HPs distributions from which draw values. 
                   These HPs preset can be specified in the '--tune-configuration' parameter.
                   Note that not all estimators can be tuned. For tabpfn a separated estimator must be used for HPs tuning.
                   Setting this parameter for untunable estimators will result in an error.""")
    
    p.add_argument("-c", "--tune-configuration", default=None,
                   help="""Tune details. It is a string representation of a dict with the following key-value couples:
                   'configuration': Name of the configuration of HPs to use. They follow the schema 'c{number}' (i.e 'c0').
                    Note that for some estimators only one configuration (c0) is available.
                    'algo': Search algorithm to use. One of 'random' and 'tpe' (default).
                    'n_iter': Number of iterations tested for the selected configuration. Must be an integer.
                    'n_repeats': Number of cv repeats used to test each sampled configuration. Must be an integer.
                    'n_splits': Number of cv splits used to test each sampled configuration. Must be an integer.
                    If None, the default, the default c0 configuration is used if "--tune" is True.
                    One can pass a partial dict using the default values for the unspecified fields.""")
    
    p.add_argument("-q", "--save-estimators", action="store_true",
                   help="""Option to save the fitted estimators. 
                   The estimators object are saved via pickle in the 'estimators' folder.
                   Note that all estimators fitted during the splitting procedure are saved.
                   The filenames follow the generic structure: {estimator}__{preprocessing}__{repetition}{fold}.
                   In case of 'holdout' splitting mode the "__{repetition}{fold}" part is replaced by
                   a sequential number.""")

    p.add_argument("--nthreads", default=16, type=int, 
                   help="Number of CPU threads to use to fit the estimators. Defaults to 16.")
    
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
        splitting_specs = try_parse_specs_into_dict(pars["splitting_specs"], "--splitting-specs")
        _check_splitting_specs_keys(splitting_mode, splitting_specs)
    
    if splitting_mode == "cv" and splitting_specs is None:
        splitting_specs = {"n_repeats": 5, "n_splits": 10}

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
