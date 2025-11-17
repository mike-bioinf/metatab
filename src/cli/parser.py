from argparse import ArgumentParser



def make_base_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)
    
    p.add_argument("-e", "--estimator", required=True, 
                    choices=["random_forest", "xgb", "es_xgb", "catboost", "es_catboost", "lgbm", "es_lgbm", "tabpfn", "autotabpfn", "finetunetabpfn"], 
                    help="""ML estimator to use. One of 'random_forest', 'xgb', 'es_xgb', 'catboost', 'es_catboost', 
                    'lgbm', 'es_lgbm', 'tabpfn', 'autotabpfn', 'finetunetabpfn'.""")
    
    p.add_argument("--early-stop-rounds", type=int, default=100,
                   help="""Number of early stop rounds to use when using the "es" estimators.
                   Must be a positive integer greater than 0. Defaults to 100.
                   This option is ignored when a non "es" estimator is used.""")

    p.add_argument("--validation-set-size", type=float, default=0.3,
                   help="""Fraction of training data to use as validation for the early stop mechanisms.
                   Must be a float in (0, 1) (0 < validation-set-size < 1).
                   This option is ignored when a non "es" estimator is used.""")

    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"], help="Defines the data input format. One between 'xy' and 'df'.")
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided when '--input-mode' is equal to 'df'")

    p.add_argument("-p", "--preprocessing", default="estimator_default", 
                    choices=["estimator_default", "base", "density_filter", "pca", "no"],
                    help= """Preprocessing to apply on the feature space. In detail:
                    -estimator_default: Automatically select from the following options the prepocessing according to the estimator used.
                    -base: Filtering of constant features.
                    -density_filter: The number of columns is reduced to 500 (approximately) keeping only the most dense features.
                    -pca: PCA preprocessing retaining the N principal components explaining the 95 percent of the variance.
                    -no: No preprocessing is applied.""")
    
    p.add_argument("--nthreads", default=16, type=int, help="Number of CPU threads to use to fit the estimators. Defaults to 16.")
    
    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p




def make_extra_fit_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)
    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    p.add_argument("-o", "--output-path", required=True, help="Path of the pickle file created in output.")
    p.add_argument("-s", "--seed", default=42, type=int, help="""Seed used to control randomness.""")
    return p




def make_extra_resample_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)

    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    
    p.add_argument("-o", "--output-dir", required=True, default=".",
                   help="""Path of the folder storing the results.
                   If not provided the folder in which the program is run is used.""")

    p.add_argument("--seed-splitter", default=42, type=int, 
                   help="Seed used to control the randomness of the splitting procedure. Defaults to 42.")
    
    p.add_argument("--seed-estimator", default=42, type=int,
                   help="""Seed used to control the randomness of the estimator.
                   This seed also controls the estimator tuning procedure when requested. Defaults to 42.""")
    
    p.add_argument("-w", "--save-realtime", action="store_true",
                   help="""Option to save the resample results in realtime. 
                   Adds a little of overhead. Highly suggested for time-consuming jobs.""")

    p.add_argument("-q", "--save-estimators", action="store_true",
                   help="""Option to save the fitted estimators. 
                   The estimators object are saved via pickle in the 'estimators' folder.
                   Note that all estimators fitted during the resample procedure are saved.
                   The filenames follow two potential generic structure:
                   - {estimator}_{repetition}{fold} when '--splitting-mode' equal 'cv';
                   - {estimator}_{number} when '--splitting-mode' equal 'holdout'.""")

    return p




def make_cv_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)

    p.add_argument("--n-cv-repeats", type=int, default=5, 
                   help="""Number of cross-validation repeats. Defaults to 5. Ignored when '--splitting-mode' equal 'holdout'.""")

    p.add_argument("--n-cv-folds", type=int, default=10, 
                   help="""Number of cross-validation folds. Defaults to 10. Ignored when '--splitting-mode' equal 'holdout'.""")

    return p




def make_holdout_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)

    p.add_argument("--n-holdout-splits", type=int, default=50, 
                    help="""Number of holdout splits. Defaults to 50. Ignored when '--splitting-mode' equal 'cv'.""")

    p.add_argument("--holdout-train-size", type=float, default=0.9, 
                   help="""Fraction of data to use for holdout training splits. Must be a positive float in (0, 1). 
                   Defaults to 0.9. Ignored when '--splitting-mode' equal 'cv'.""")

    return p




def make_tune_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)

    p.add_argument("--tune-algo", choices=["random", "tpe", "meta"], default="tpe",
                   help="""Optimization algorithm to use. Possible options are 'random', 'tpe'(default) and 'meta'.
                   The meta option enables a metalearning powered tuning, where the points are suggested
                   by a surrogate model trained on our tuning prior.Three important notes:
                   1. The tuning prior is generated on a collection of 32 real datasets (info here).
                   Therefore this strategy should NOT be used on these datasets to avoid leakage and overoptimistic results.
                   2. The tuning prior is generate only considering the default '--tune-space' for every estimator.
                   An error will be raised if an alternative space is requested.
                   3. Is highly suggested to use the 'estimator_default' preprocessing when meta optimizing 
                   since the tuning prior has been generated only considering this option. 
                   Selecting a different preprocessing can greatly hurt performance. """)
    
    # TODO: add reference to spaces
    p.add_argument("--tune-space", default="default", 
                   help="""Pre-defined HPs space to use. They follow the schema 'c{number}' (i.e 'c0').
                   The wildcard 'default' can be used to select the default one for every estimator.""")
    
    p.add_argument("--tune-n-iter", type=int, default=100, 
                   help="""Number of points to evaluate in the tune procedure. Defaults to 100.
                   When '--tune-algo' option equal 'tpe' is suggested to use values >= 30. 
                   This is because the tpe procedure is set with a fixed warm-up phase of 20 iterations. 
                   So when 20 or less iterations are given the tpe procedure becomes a random one.""")

    p.add_argument("--tune-n-cv-repeats", type=int, default=1, 
                   help="""Number of times to repeat the inner cross-validation used during hyperparameter tuning. 
                   Repeating CV reduces the variance of tuning scores but increases training time. Defaults to 1.""")

    p.add_argument("--tune-n-cv-folds", type=int, default=5, 
                   help="""Number of folds used in the inner cross-validation during hyperparameter tuning. Defaults to 5.""")
    
    p.add_argument("--tune-meta-surrogate-model", default=None,
                   help="""Not intended for users. This parameter allows to pass an external surrogate model 
                   to use instead of the default one in the tune meta optimization process. The model must be serialized with joblib.
                   Ignored when '--tune-algo' is not "meta".""")
    
    # TODO: for now we do not allow to specify the specifics for "random_from_best" and "uniform_from_best" strategies
    p.add_argument("--tune-meta-strategy", choices=["best", "random_from_best", "uniform_from_best"], default="random_from_best",
                   help="""Strategy used to select the points proposed and evaluated by the surrogate model.
                   These points are the ones that will be tested in the inner cv.
                   -best: The n '--tune-n-iter' best points according to the surrogate model are selected.
                   -random_from_best: '--tune-n-iter' points are selected randomly from the best.
                   -uniform_from_best: '--tune-n-iter' points are selected with a fixed step size from the best.
                   Ignored when the '--tune-algo' is not "meta".""")
    
    return p