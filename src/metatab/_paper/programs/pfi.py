"""
Program to compute feature importance score based on a permutation approach.
The program process all permutations associated to one feature in one step.
Optionally is possible also to process multiple feature together using the '--permutation-batch-size' parameter.
This could speed up the process especially for gpu based models even though raises memory peak.
Feature importance is quantified through the increase in logloss associated to a permutation.
Negative scores (better logloss after pemutation) have no real meaning but possible due to statistical fluctations.
"""

import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from metatab.cli.helper import create_logger
from metatab.estimators.pycore.estimators import *
from shutil import rmtree



def parse_args(args):
    p = ArgumentParser()
    p.add_argument("--dataset-path", required=True, help="path to the dataset file")
    p.add_argument("--output-folder", required=True, help="Path to the output folder")
    p.add_argument(
        "--estimator", required=True, 
        choices=["random_forest", "extra_trees", "catboost", "xgb", "lgbm", "realmlp", "tabm", "tabpfn", "es_xgb", "es_lgbm", "es_catboost"], 
        help="Classifier to use"
    )
    p.add_argument("--estimator-mode", default="default", choices=["default", "tune", "ensemble"], help="How to train the classifier")
    p.add_argument("--y", default="Group", help="Name of the target column")
    p.add_argument("--n-permutation", default=20, type=int, help="Number of permutation to use to compute feature importance scores")
    p.add_argument("--permutation-batch-size", default=20, type=int, help="Number of feature whose permutations are processed together")
    p.add_argument("--n-cv-repeat", default=3, type=int, help="Number of cross validation repeats")
    p.add_argument("--n-cv-fold", default=5, type=int, help="Number of cross validation folds")
    p.add_argument("--seed", default=0, type=int, help="Seed to control randomness")
    p.add_argument("--n-threads", default=8, type=int, help="Number of threads to use")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device on which to run the estimators")
    return p.parse_args(args)



def dispatch_estimator(estimator: str, estimator_mode: str):
    estimators = {
        "catboost": {
            "default": DefaultCatBoostClassifier,
            "tune": StandardTuneCatBoostClassifier,
            "ensemble": StandardEnsembleCatBoostClassifier,
        },
        "xgb": {
            "default": DefaultXGBClassifier,
            "tune": StandardTuneXGBClassifier,
            "ensemble": StandardEnsembleXGBClassifier,
        },
        "lgbm": {
            "default": DefaultLGBMClassifier,
            "tune": StandardTuneLGBMClassifier,
            "ensemble": StandardEnsembleLGBMClassifier,
        },
        "random_forest": {
            "default": DefaultRandomForestClassifier,
            "tune": StandardTuneRandomForestClassifier,
            "ensemble": StandardEnsembleRandomForestClassifier,
        },
        "extra_trees": {
            "default": DefaultExtraTreesClassifier,
            "tune": StandardTuneExtraTreesClassifier,
            "ensemble": StandardEnsembleExtraTreesClassifier,
        },
        "realmlp": {
            "default": DefaultRealMLPClassifier,
            "tune": StandardTuneRealMLPClassifier,
            "ensemble": StandardEnsembleRealMLPClassifier,
        },
        "tabm": {
            "default": DefaultTabMClassifier,
            "tune": StandardTuneTabMClassifier,
            "ensemble": StandardEnsembleTabMClassifier,
        },
        "tabpfn": {
            "default": DefaultTabPFNClassifier,
            "tune": StandardTuneTabPFNClassifier,
            "ensemble": StandardEnsembleTabPFNClassifier,
        },
        "es_xgb": {
            "default": DefaultEsXGBClassifier,
            "tune": StandardTuneEsXGBClassifier,
            "ensemble": StandardEnsembleEsXGBClassifier,
        },
        "es_lgbm": {
            "default": DefaultEsLGBMClassifier,
            "tune": StandardTuneEsLGBMClassifier,
            "ensemble": StandardEnsembleEsLGBMClassifier,
        },
        "es_catboost": {
            "default": DefaultEsCatBoostClassifier,
            "tune": StandardTuneEsCatBoostClassifier,
            "ensemble": StandardEnsembleEsCatBoostClassifier,
        },
    }

    if estimator not in estimators:
        raise ValueError(f"Unsupported estimator: {estimator}")

    if estimator_mode not in estimators[estimator]:
        raise ValueError(f"Unsupported estimator mode: {estimator_mode}")

    return estimators[estimator][estimator_mode]


def binary_logloss(y_true: np.ndarray, predictions: np.ndarray, eps=1e-15) -> np.ndarray:
    '''
    Compute element-wise binary log-loss.
    Inputs must have broadcast-compatible shapes.
    '''
    # we convert to float64 to avoid "clipping failures" with larger eps
    p = np.clip(predictions.astype(np.float64), eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log1p(-p))


def get_estimator_mode_specs(estimator_mode: str, output_path: Path) -> dict:
    '''
    Returns a dict of specs specific to the training regime to follow in training.
    '''
    if estimator_mode == "default":
        return {}
    elif estimator_mode == "tune":
        return {"n_iter": 20, "algo": "random"}
    elif estimator_mode == "ensemble":
        return {"save_path": output_path / "ensemble"}
    else:
        return ValueError(f"Unsupported estimator mode: {estimator_mode}")



def main():
    pars = vars(parse_args(sys.argv[1:]))
    logger = create_logger(sys.stdout)
    logger.info(f"Launching PFI program on dataset '{pars["dataset_path"]}' with classifier '{pars["estimator"]}'")
    
    # load data
    data = pd.read_csv(pars["dataset_path"], sep="\t")
    X = data.drop(columns=[pars["y"]])
    y: pd.Series = data[pars["y"]]
    logger.info("Data loaded in memory")
    
    # create output folder
    output_folder = Path(pars["output_folder"])
    output_folder.mkdir(exist_ok=True, parents=True)

    splitter = RepeatedStratifiedKFold(
        n_splits=pars["n_cv_fold"], 
        n_repeats=pars["n_cv_repeat"], 
        random_state=pars["seed"]
    )

    cls_args = get_estimator_mode_specs(pars["estimator_mode"], output_folder)
    n_permutations = pars["n_permutation"]
    permutation_batch_size = pars["permutation_batch_size"]
    cls_class = dispatch_estimator(pars["estimator"], pars["estimator_mode"])
    dfs_pfi = []
    
    # we remove 0 variance features to not permute them in the test set
    # NOTE: we don't account for the feature filtered for the early-stopped estimators
    # This does not corrupt the results since the importance 
    # of the non-filtered features will be 0 but we waste computation
    vt = VarianceThreshold().set_output(transform="pandas")

    # cross validation
    for iteration, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat = iteration // pars["n_cv_fold"]
        fold = iteration - (pars["n_cv_fold"] * repeat)
        logger.info(f"Starting fold {fold} of repeat {repeat}")
        
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]
        
        X_train_sel = vt.fit_transform(X_train)
        X_test_sel = vt.transform(X_test)

        cls = cls_class(
            preprocessing="base",
            device=pars["device"],
            seed = pars["seed"], 
            n_threads = pars["n_threads"],
            **cls_args
        )

        cls.fit(X_train_sel, y_train)

        n_samples, n_features = X_test_sel.shape
        rng = np.random.default_rng(pars["seed"] + iteration)

        # Baseline predictions
        baseline_pred = cls.predict_proba(X_test_sel)[:, 1]
        baseline_loss = binary_logloss(y_test, baseline_pred).mean()

        # Shape: (n_permutations, n_features)
        importances = np.zeros((n_permutations, n_features))

        # Process multiple features in each batch
        seq_start_batches = list(range(0, n_features, permutation_batch_size))
        n_batches = len(seq_start_batches)

        for n_batch, start in enumerate(seq_start_batches):
            end = min(start + permutation_batch_size, n_features)
            batches = []
            batch_info = []

            for feature_idx in range(start, end):
                original_values = X_test_sel.iloc[:, feature_idx].to_numpy()

                for p in range(n_permutations):
                    X_perm = X_test_sel.copy()
                    X_perm.iloc[:, feature_idx] = original_values[rng.permutation(n_samples)]
                    batches.append(X_perm)
                    batch_info.append((p, feature_idx))

            # Concatenate all permuted datasets
            X_batch = pd.concat(batches, ignore_index=True)

            # One prediction call for the entire batch
            predictions = cls.predict_proba(X_batch)[:, 1].astype(np.float64)

            # Shape: (feature, permutation, sample)
            predictions = predictions.reshape(end - start, n_permutations, n_samples)

            # Shape: (1, 1, sample)
            y_true = np.asarray(y_test)[None, None, :]

            # compute loss
            losses = binary_logloss(y_true, predictions)

            # compute mean loss across samples --> (Feature, Permutation)
            permuted_losses = losses.mean(axis=2)
            importances[:, start:end] = (permuted_losses - baseline_loss).T
            logger.info(f"Processed batch {n_batch} of {n_batches - 1}")

        # Restore original feature space
        df_pfi = pd.DataFrame(importances, columns=X_test_sel.columns)
        df_pfi = df_pfi.reindex(columns=X.columns,fill_value=0.0)
        df_pfi["repeat"] = repeat
        df_pfi["fold"] = fold
        dfs_pfi.append(df_pfi)


    # concat and save
    pd.concat(dfs_pfi, axis=0, ignore_index=True).to_csv(
        output_folder / f"pfi.txt",
        sep="\t",
        index=False
    )

    # clean ensemble models
    if pars["estimator_mode"] == "ensemble":
        rmtree(output_folder / "ensemble")

    logger.info(f"Output created at {output_folder}")

    


if __name__ == "__main__":
    main()