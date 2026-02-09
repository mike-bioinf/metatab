from metatab.cli.helper import h
from argparse import ArgumentParser, RawTextHelpFormatter


def make_base_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    
    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file. See 'input_mode' for details.")
    
    p.add_argument("-o", "--output-dir", required=True, help="Path of the folder where the results are created.")

    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"], 
                   help=h("""
                    Defines the data input format. One between 'xy' and 'df'.
                    -xy: A folder containing `X.txt` and `y.txt` files. The file names as well as the extensions must exactly match. 
                    -df: A file containing both X and y features in a table.
                    In both cases the program DEMANDS tab-separated text files."""))
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided when '--input-mode' is equal to 'df'")
    
    p.add_argument("--nthreads", default=16, type=int, help="Number of CPU threads to use to fit the estimator(s). Defaults to 16.")
    
    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p



# set of options that are NOT used by family_ensemble and autogluon modes
def make_extra_base_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("-e", "--estimator", required=True,
                    choices=[
                        "random_forest", "extra_trees", "xgb", "es_xgb", "catboost", "es_catboost", 
                        "lgbm", "es_lgbm", "tabpfn", "realmlp", "tabm"
                    ],
                    help=h("""
                    Machine learning classifier to use.
                    For GBDTs classifiers is possible to select the 'es_*' versions which use early stop on validation set. 
                    In this case one can adjust the '--early-stop-rounds' and "--validation-set-size" parameters to control the early stop behaviour.
                    Be mindful that 'realmlp' and 'tabm' estimators always uses early stop on validation set.
                    In this case is not possible to suppress this learning behaviour. 
                    In addition they use advanced early stop mechanisms which do not rely on a fixed number of rounds,
                    therefore the '--early-stop-rounds' parameter has no effect."""))
    
    p.add_argument("--early-stop-rounds", type=int, default=100,
                   help=h("""
                   Number of early stop rounds for "es" GBDT models only (es_catboost, es_xgb and es_lgbm).
                   Must be a positive integer greater than 0. Defaults to 100.
                   This option is ignored when a non early stoppable estimator is used.
                   Note: ignored by "realmlp" and "tabm" estimators also since they use more sophisticaed early stop strategies.
                   Note: the usage of this parameter is discouraged since it can be removed in future versions."""))

    p.add_argument("--validation-set-size", type=float, default=0.3,
                   help=h("""
                   Fraction of training data to use as validation for early stop. Must be a float in (0, 1).
                   This option is ignored when a non early stoppable estimator is used."""))

    p.add_argument("-p", "--preprocessing", default="estimator_default", 
                    choices=["estimator_default", "base", "density_filter", "pca", "no"],
                    help=h("""
                    Data preprocessing strategy:
                    -estimator_default: Automatically select one of following options according to the estimator used.
                    -base: Filtering of constant features.
                    -density_filter: The number of columns is reduced to 500 (approximately) keeping only the most dense features.
                    -pca: PCA preprocessing retaining the N principal components explaining the 95 percent of the variance.
                    -no: No preprocessing is applied."""))
    
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                    help=h("""
                    Select the device on which fit the estimator(s). 
                    When auto (default) 'cuda' is selected if available AND the estimator requires GPU else 'cpu'.
                    Note: The GBDTs estimators can be run only on CPU."""))
    
    return p




def make_fit_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    p.add_argument("-s", "--seed", default=42, type=int, help="""Seed used to control randomness.""")
    return p




def make_base_resample_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("--seed-splitter", default=42, type=int, 
                   help="Seed used to control the randomness of the splitting procedure. Defaults to 42.")

    p.add_argument("--save-estimators", action="store_true",
                   help=h("""
                   Option to save the fitted estimators. 
                   The estimators object are saved via pickle in the 'estimators' folder.
                   Note that all estimators fitted during the resample procedure are saved.
                   The filenames follow two potential generic structure:
                   - {estimator}_{repeat}{fold} with the 'metatab-resample cv' subcommand;
                   - {estimator}_{number} with the 'metatab-resample holdout' subcommand."""))

    p.add_argument("--disable-additional-txt-output", action="store_true", 
                   help=h("""
                   Disable the generation of the txt files of the predicted estimator probabilities, original and encoded target values.
                   In this case these info are only available in the encoded str format in the main output."""))
    return p



def make_resample_seed_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("--seed-estimator", default=42, type=int, 
                   help="""Seed used to control the estimators randomness. Defaults to 42.""")

    return p




def make_cv_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("--n-cv-repeats", type=int, default=5, 
                   help="""Number of cross-validation repeats. Defaults to 5.""")

    p.add_argument("--n-cv-folds", type=int, default=10, 
                   help="""Number of cross-validation folds. Defaults to 10.""")

    return p




def make_holdout_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("--n-holdout-splits", type=int, default=50, 
                    help="""Number of holdout splits. Defaults to 50.""")

    p.add_argument("--holdout-train-size", type=float, default=0.9, 
                   help="""Fraction of data to use for holdout training splits. Must be a positive float in (0, 1). Defaults to 0.9.""")

    return p




def make_tune_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    # TODO: add link where to find info
    p.add_argument("--tune-algo", choices=["random", "tpe", "meta"], default="tpe",
                   help=h("""
                    Optimization algorithm to use. Possible options are 'random', 'tpe'(default) and 'meta'.
                   The meta option enables a metalearning powered tuning, where the points are suggested
                   by a surrogate model trained on our tuning prior.Three important notes:
                   1. The tuning prior is generated on a collection of 32 real datasets (see paper for details).
                   Therefore this strategy should NOT be used on these datasets to avoid leakage and overoptimistic results.
                   2. The tuning prior is generate only considering the default '--tune-space' for every estimator.
                   An error will be raised if an alternative space is requested.
                   3. Is highly suggested to use the 'estimator_default' preprocessing when meta optimizing 
                   since the tuning prior has been generated only considering this option. 
                   Selecting a different preprocessing can greatly hurt performance."""))
    
    # TODO: add reference to spaces
    p.add_argument("--tune-space", default="default", 
                   help=h(
                    """Pre-defined HPs space to use. They follow the schema 'c{number}' (i.e 'c0').
                   The wildcard 'default' can be used to select the default one for every estimator."""))
    
    p.add_argument("--tune-n-iter", type=int, default=100, 
                   help=h("""
                   Number of points to evaluate in the tune procedure. Defaults to 100.
                   When '--tune-algo' option equal 'tpe' is suggested to use values >= 30. 
                   This is because the tpe procedure is set with a fixed warm-up phase of 20 iterations. 
                   So when 20 or less iterations are given the tpe procedure becomes a random one."""))

    p.add_argument("--tune-n-cv-repeats", type=int, default=1, 
                   help=h("""
                   Number of times to repeat the inner cross-validation used during hyperparameter tuning. 
                   Repeating CV reduces the variance of tuning scores but increases training time. Defaults to 1."""))

    p.add_argument("--tune-n-cv-folds", type=int, default=5, 
                   help="""Number of folds used in the inner cross-validation during hyperparameter tuning. Defaults to 5.""")
    
    p.add_argument("--tune-meta-surrogate-model", default=None,
                   help=h("""
                   For advanced usage only. 
                   This parameter allows to pass an external surrogate model to use in the tune meta optimization process.
                   Ignored when '--tune-algo' is not "meta"."""))
    
    # TODO: for now we do not allow to specify the specifics for the strategies
    p.add_argument("--tune-meta-strategy", choices=["best", "random_from_best", "uniform_from_best", "random_uniform_from_best"], default="best",
                   help=h("""
                    Strategy used to select the points evaluated and proposed by the meta-framework.
                   These points are the ones that will be tested in the inner cv.
                   -best: The n '--tune-n-iter' best points according to the surrogate model are selected.
                   -random_from_best: '--tune-n-iter' points are selected randomly from the best.
                   -uniform_from_best: '--tune-n-iter' points are selected with a fixed step size from the best.
                   -random_uniform_from_best: '--tune-n-iter' points are selected randomly in intervals defined with a fixed step size from the best.
                   Ignored when the '--tune-algo' is not "meta"."""))
    
    return p




def make_ensemble_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("--ensemble-name", default="ens", 
                   help="""Ensemble name. The ensemble members are nominated as '{name}_m{number}'.""")

    # TODO: add link where to find info
    p.add_argument("--ensemble-algo", choices=["random", "meta"], default="meta",
                    help=h("""
                    How to derive the hps configurations to ensemble.
                    The meta option enables a metalearning powered procedure, where the points are suggested
                    by a surrogate model trained on our tuning prior. Three important notes:
                    1. The tuning prior is generated on a collection of 32 real datasets (see paper for details).
                    Therefore this strategy should NOT be used on these datasets to avoid leakage and overoptimistic results.
                    2. The tuning prior is generate only considering the default '--ensemble-space' for every estimator.
                    An error will be raised if an alternative space is requested.
                    3. Is highly suggested to use the 'estimator_default' preprocessing when meta optimizing 
                    since the tuning prior has been generated only considering this option. 
                    Selecting a different preprocessing can greatly hurt performance."""))
    
    # TODO: add reference to spaces
    p.add_argument("--ensemble-space", default="default",
                   help=h("""
                   Pre-defined HPs space to use. They follow the schema 'c{number}' (i.e 'c0').
                   The wildcard 'default' can be used to select the default one for every estimator."""))

    p.add_argument("--ensemble-n-members", default=16, type=int, help="Number of ensemble members.")

    p.add_argument("--ensemble-time-limit", type=int, default="10000000", help="Time limit for ensembling.")

    p.add_argument("--ensemble-meta-surrogate-model", default=None,
                   help=h("""
                   For advanced usage only. This parameter allows to pass an external surrogate model to use in the meta-ensembling process.
                   Ignored when '--ensemble-algo' is not "meta"."""))
    
    # TODO: for now we do not allow to specify the specifics for the strategies
    p.add_argument("--ensemble-meta-strategy",
                   choices=["best", "random_from_best", "uniform_from_best", "random_uniform_from_best"], 
                   default="random_uniform_from_best",
                   help=h("""
                    Strategy used to select the hps points evaluated by the surrogate model.
                   -best: The n 'ensemble-n-members' best points according to the surrogate model are selected.
                   -random_from_best: 'ensemble-n-members' points are selected randomly from the best.
                   -uniform_from_best: 'ensemble-n-members' points are selected with a fixed step size from the best.
                   -random_uniform_from_best: '--ensemble-n-iter' points are selected randomly in intervals defined with a fixed step size from the best.
                   Ignored when the '--ensemble-algo' is not "meta"."""))
    
    return p




def make_family_ensemble_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    
    p.add_argument("--ensemble-name", default="family_ensemble", help="Family ensemble name.")

    p.add_argument("--ensemble-configuration", default="all_meta_16",
                   help=h("""
                    Family ensemble configuration. Accepts wildcards or a json configuration file.
                   The wildcard options comes from the pattern (all|cpu|gpu)_(meta|random)_{n_members} where:
                   - all|cpu|gpu: set the estimators. "all" selects all estimators, "cpu" only the
                   cpu-main-based ones, and "gpu" only the gpu-main-based ones.
                   - meta|random: The algorithm used to infer the estimators hps configurations.
                   - n_members: The number of members for every inner estimator ensemble.
                   In this case the estimators that can be GPU-accelerated are run on GPU if CUDA is available.
                   If a json file it must be derived from the "CollectionUserEnsembleConfiguration" 
                   python class or must follow its signature."""))

    p.add_argument("--use-bag-cv", action="store_true",
                   help=h("""
                    Flag to enable a bagged cross-validation approach to subset the input data
                   in the ensembling approach. One must specify the number of repeats and folds to use (see following options). 
                   It's important to note that the number of total iterations (repeats * folds) must be greater or equal than 
                   the number of estimators used."""))

    p.add_argument("--bag-cv-repats", type=int, default=5, 
                   help="Number of repeats to use in the bag-cv procedure. Ignored when '--use-bag-cv' flag is down.")

    p.add_argument("--bag-cv-folds", type=int, default=5, 
                   help="Number of folds to use in the bag-cv procedure. Ignored when '--use-bag-cv' flag is down.")

    p.add_argument("--feature-space-randomization", type=float, default=1, 
                   help="Set the proportion of features space randomly selected for every estimator.")

    p.add_argument("--ensemble-time-limit", type=int, default="10000000", help="Time limit for ensembling.")

    return p




def make_autogluon_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)

    p.add_argument("--preset", default="extreme_quality", 
                   help=h("""
                    Autogluon preset to use. Set the ML classifiers used by autogluon and the relative hps space portfolio.
                   See "https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.fit.html" for additional info."""))

    p.add_argument("--time-limit", default=60, type=int, help="Fit time limit.")
    
    p.add_argument("--n-columns-density-filter", default=500, type=int,
                   help=h("""
                   Filter the most sparse columns in the data to reach the specified number of columns.
                   Ties are arbitrarily broken in a reproducible way thanks to an underlying sorting stable algorithm.
                   Keep in mind that changing the feature order can affect the filtering since ties will be internally ordered in a different way.
                   Useful to allow autogluon to fit tabpfn and other foundational models with a limited feature window.
                   The default value of 500 is choosen based on the feature limit window of tabpfn and mitra models.
                   If a number greater than the actual number of columns is used, then all columns are selected (no filtering).
                   Note: autogluon skip the foundational models when the number of features is not within their feature limits. 
                   Note that these models are run only with the most performant presets. 
                   So with lower quality presets the filtering process can be skipped.
                   See autogluon documentation for detailed info about the presets."""))
    
    p.add_argument("--eval-metric", default="log_loss",
                   help=h("""
                   Name of the validation metric used by autogluon. 
                   See "https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html for all options."""))

    p.add_argument("--ngpus", default=0, type=int,
                   help="Number of gpus to use. It fallbacks to the available ones when a number greater than that is used.")
    
    return p