import argparse


def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-i", "--input-data", required=True, help="Path to the dataset folder/file.")
    p.add_argument("-o", "--output-path", required=True, help="Path of the pickle file created in output.")

    p.add_argument("-m", "--input-mode", required=True, choices=["sets", "xy", "df"],
                    help="Defines the data input format. One of 'sets', 'xy', or 'df'.")

    p.add_argument("-e", "--estimator", required=True, 
                    choices=["random_forest", "xgb", "es_xgb", "tabpfn"], 
                    help="""ML 'estimator' to use. One of 'random_forest', 'xgb', 'es_xgb', 'tabpfn'.""")
    
    p.add_argument("-p", "--preprocessing", default="base", choices=["base", "density_filter", "pca"],
                    help= """Preprocessing to apply on the feature space. One of 'base', 'density_filter' and 'pca'.
                    -base: a general minimal preprocessing is applied according to the used estimator.
                    -density_filter: The number of columns is reduced to 500 keeping only the most dense features.
                    Note: according to the type of filtering applied the exact number of selected features may be not exactly 500. 
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

    p.add_argument("--create-outdir", action="store_true", help="Create the output folder if does not exists.")

    return p.parse_args(args)

