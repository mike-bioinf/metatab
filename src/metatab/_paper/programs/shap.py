import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from shap import TreeExplainer

from metatab.estimators.pycore.estimators import (
    DefaultCatBoostClassifier,
    DefaultLGBMClassifier,
    DefaultXGBClassifier,
    DefaultExtraTreesClassifier,
    DefaultRandomForestClassifier,
)



def parse_args(args):
    p = ArgumentParser()
    p.add_argument("--dataset-path", required=True, help="path to the dataset file")
    p.add_argument("--output-folder", required=True, help="Path to the output folder")
    p.add_argument("--estimator", required=True, choices=["randomforest", "extratrees", "catboost", "xgb", "lgbm"], help="Classifier to use")
    p.add_argument("--y", default="Group", help="Name of the target column")
    p.add_argument("--n-cv-repeat", default=3, type=int, help="Number of cross validation repeats")
    p.add_argument("--n-cv-fold", default=5, type=int, help="Number of cross validation folds")
    p.add_argument("--seed", default=0, type=int, help="Seed to control randomness")
    p.add_argument("--n-threads", default=8, type=int, help="Number of threads to use")
    return p.parse_args(args)



def dispatch_estimator(estimator: str):
    match estimator:
        case "catboost":
            return DefaultCatBoostClassifier
        case "xgb":
            return DefaultXGBClassifier
        case "lgbm":
            return DefaultLGBMClassifier
        case "randomforest":
            return DefaultRandomForestClassifier
        case "extratrees":
            return DefaultExtraTreesClassifier
        case _:
            return ValueError(f"Unsupported estimator: {estimator}")



def main():
    pars = vars(parse_args(sys.argv[1:]))
    
    # load data
    data = pd.read_csv(pars["dataset_path"], sep="\t")
    X = data.drop(columns=[pars["y"]])
    y: pd.Series = data[pars["y"]]
    
    # create output folder
    output_folder = Path(pars["output_folder"])
    output_folder.mkdir(exist_ok=True, parents=True)

    splitter = RepeatedStratifiedKFold(
        n_splits=pars["n_cv_fold"], 
        n_repeats=pars["n_cv_repeat"], 
        random_state=pars["seed"]
    )

    cls_class = dispatch_estimator(pars["estimator"])
    # metatab uses by default a zero-variance filtering.
    # we explicitely account for this to avoid shap computation errors
    vt = VarianceThreshold().set_output(transform="pandas")

    # cross validation
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat = i // pars["n_cv_fold"]
        fold = i - (pars["n_cv_fold"] * repeat)
        
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]
        
        X_train_sel = vt.fit_transform(X_train)
        X_test_sel = vt.transform(X_test)

        cls = cls_class(seed = pars["seed"], n_threads = pars["n_threads"], preprocessing="base")
        cls.fit(X_train_sel, y_train)

        explainer = TreeExplainer(
            cls.estimator_.estimator_[-1],
            data=X_train_sel,
            model_output="probability",
            feature_perturbation="interventional"
        )

        shap_values = explainer(X_test_sel)
        
        # we select shap referred to class 1:
        # for rf and extratrees we have shapr for both class 0 and 1 
        # but they are the same number with opposite sign
        if shap_values.values.ndim == 3:
            values = shap_values.values[:, :, 1]
        else:
            values = shap_values.values

        df_shap = pd.DataFrame(values, index=X_test.index, columns=X_train_sel.columns)

        # reintroduce removed feature with SHAP equal to 0
        df_shap = df_shap.reindex(columns=X.columns, fill_value=0.0)
        df_shap.to_csv(output_folder / f"shap_{repeat}{fold}.txt", index=True)
    




if __name__ == "__main__":
    main()