import argparse


def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    p.add_argument("-o", "--output-path", required=True, help="Path of the pickle file created in output.")

    p.add_argument("-m", "--input-mode", required=True, choices=["sets", "xy", "df"],
                    help="Defines the data input format. One of 'sets', 'xy', or 'df'.")

    p.add_argument("-e", "--estimator", required=True, 
                    choices=["random_forest", "xgb", "es_xgb", "catboost", "es_catboost", "lgbm", "es_lgbm", "tabpfn", "autotabpfn", "finetunetabpfn"], 
                    help="""ML estimator to use. One of 'random_forest', 'xgb', 'es_xgb', 'catboost', 'es_catboost', 
                    'lgbm', 'es_lgbm', 'tabpfn', 'autotabpfn', 'finetunetabpfn'.""")
    
    p.add_argument("-r", "--early-stopping-rounds", type=int, default=-1,
                   help="""Number of early stop rounds to use when using the "es" estimators.
                   The default is -1, which means 100 for the "es" estimators and a value to be
                   ignored by the "non es" estimators (in this case other values will results in an error).""")
    
    p.add_argument("-y", "--target-feature", default=None, 
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")
    
    p.add_argument("-p", "--preprocessing", default="estimator_default", 
                    choices=["estimator_default", "base", "density_filter", "pca", "no"],
                    help= """Preprocessing to apply on the feature space. In detail:
                    -estimator_default: Automatically select from the following options the prepocessing according to the estimator used.
                    -base: Filtering of constant features.
                    -density_filter: The number of columns is reduced to 500 (approximately) keeping only the most dense features.
                    -pca: PCA preprocessing retaining the N principal components explaining the 95 percent of the variance.
                    -no: No preprocessing is applied.""")
    
    p.add_argument("-t", "--tune", action="store_true", 
                   help="""Tune the estimator hyperparameters. 
                   We use presets of HPs distributions from which draw values. These presets can be specified 
                   in the '--tune-configuration' parameter. Note that not all estimators can be tuned. 
                   For tabpfn a separated estimator must be used for HPs tuning.
                   Setting this parameter for untunable estimators will result in an error.""")
    
    p.add_argument("-c", "--tune-configuration", default=None,
                    help="""Tune details. It is a string representation of a dict with the following key-value couples:
                   'configuration': Name of the space of HPs to use. They follow the schema 'c{number}' (i.e 'c0').
                    Note that for some estimators only one space (c0) is available.
                    Is also possible to use the selected default space of each estimator using the value "default".
                    'algo': Search algorithm to use. One of 'random', 'tpe' (default) and 'meta' (works only with the default configuration).
                    'n_iter': Number of iterations tested for the selected configuration. Must be an integer.
                    'n_repeats': Number of cv repeats used to test each sampled configuration. Must be an integer.
                    'n_splits': Number of cv splits used to test each sampled configuration. Must be an integer.
                    If None, the default, a default configuration is used when "--tune" is True.
                    One can pass a partial dict using the default values for the unspecified fields.""")

    p.add_argument("-s", "--seed", default=42, type=int, 
                   help="""Seed used to control randomness.
                   In particular it controls the randomness inherent to the estimators, splitting and tuning procedures.""")

    p.add_argument("--meta-database", default=None,
                   help="""Not intended for users. This parameter allows to pass as a str dict representation
                   "'estimator':'path_model'" the database of surrogate models to use in the metalearning scenarios.
                   This override the internal one, which is used instead when the argument is None.""")

    p.add_argument("--nthreads", default=16, type=int, 
                   help="Number of CPU threads to use to fit the estimator. Defaults to 16.")
    
    p.add_argument("--create-outdir", action="store_true", help="Create the output folder if does not exists.")

    return p.parse_args(args)

