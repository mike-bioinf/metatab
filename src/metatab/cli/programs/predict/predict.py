"""
Program to do inference on a dataset using an serialized fitted model via the "fit" program.
"""

import sys
import pickle
import numpy as np
from autogluon.tabular import TabularPredictor
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.prediction import PredictionDataframe
from metatab.estimators.estimators import Estimator
from metatab.ensemble.family import FamilyEnsembleEstimator

from metatab.cli.helper import (
    adjust_io_paths_, 
    manage_output_path,
    check_target_feature, 
    create_logger
)

from metatab.cli.programs.predict.helper import (
    parse_args,
    check_is_estimator_object,
    check_estimator_is_fitted
)



def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_target_feature(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    logger = create_logger(sys.stdout)

    # deserialize estimator
    with open(pars["file_estimator"], "rb") as f:
        estimator: Estimator = pickle.load(f)

    check_is_estimator_object(estimator)
    check_estimator_is_fitted(estimator)
    logger.debug("Estimator deserialized!")

    dl = DataLoader()

    # load data
    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="test",
        skip=["X_train", "y_train"]
    )

    X, y = dl.X_test, dl.y_test
    logger.debug("Data loaded in memory!")

    # here we consider all 3 load methods to retrieve the predict dataset name
    predict_dataset_name = dl.test_dataset_name if dl.test_dataset_name else dl.generic_dataset_name
    fit_dataset_name = estimator._fit_dataset_name_

    # encode y
    y_enc = estimator._le_.transform(y)

    if isinstance(estimator, (FamilyEnsembleEstimator, TabularPredictor)):
        fit_preprocessing_dict = {}
    else:
        fit_preprocessing_dict = estimator.collect_fit_preprocessing_info()
    
    # uniform feature space when requested
    if pars["x_uniform"]:
        fit_features = estimator._fit_features_
        if not np.isin(X.columns.to_numpy(), fit_features).any():
            raise ValueError(
                "Test feature space has no feature in common with the training space."
            )
        X = X.reindex(columns=fit_features, fill_value=0.0)
        logger.debug("Test feature space uniformed to training space.")
    
    if isinstance(estimator, TabularPredictor):
        pred_proba = estimator.predict_proba(X, as_pandas=False)
        preprocessing = None
    else:
        pred_proba = estimator.predict_proba(X)
        preprocessing = estimator.preprocessing

    pdf = PredictionDataframe()

    pdf.build_from_data(
        dataset=fit_dataset_name,
        y_test=y_enc,
        pred_proba=pred_proba,
        classes=estimator._classes_,
        classes_counts=estimator._classes_counts_,
        save_path=None,
        predict_dataset=predict_dataset_name,
        preprocessing=preprocessing,
        **fit_preprocessing_dict
    )
    
    pdf.compute_metrics(multiclass="average", average_strategy="macro")
    filename = f"pred_df__{fit_dataset_name}__{predict_dataset_name}.txt"
    filepath = pars["output_dir"] / filename
    pdf.to_csv(filepath, sep="\t", index=False)
    logger.debug(f"Output created at: {filepath}.")



if __name__ == "__main__":
    main()