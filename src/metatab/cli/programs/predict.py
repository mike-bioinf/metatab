"""
Program to do inference on a dataset using a pickled serialized fitted estimator.
"""

import sys
import pickle
import argparse
import numpy as np
from metatab.cli.helper import h
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.prediction.dataframe import PredictionDataframe
from metatab.estimators.estimators import Estimator
from metatab.ensemble.family import FamilyEnsembleEstimator
from autogluon.tabular import TabularPredictor

from metatab.cli.helper import (
    adjust_io_paths_, 
    manage_output_path,
    check_target_feature, 
    create_logger
)

from metatab.estimators.estimators import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyEnsembledRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyEnsembledXGBClassifier,
    MyEnsembledESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyEnsembledCatBoostClassifier,
    MyEnsembledESCatBoostClassifier,
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyEnsembledLGBMClassifier,
    MyEnsembledESLGBMClassifier,
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    MyEnsembledTabPFNClassifier
)



def check_is_estimator_object(obj) -> None:
    if not isinstance(
        obj, 
        (
            MyRandomForestClassifier,
            MyTunedRandomForestClassifier,
            MyEnsembledRandomForestClassifier,
            MyXGBClassifier,
            MyESXGBClassifier,
            MyTunedXGBClassifier,
            MyTunedESXGBClassifier,
            MyEnsembledXGBClassifier,
            MyEnsembledESXGBClassifier,
            MyCatBoostClassifier,
            MyESCatBoostClassifier,
            MyTunedCatBoostClassifier,
            MyTunedESCatBoostClassifier,
            MyEnsembledCatBoostClassifier,
            MyEnsembledESCatBoostClassifier,
            MyLGBMClassifier,
            MyESLGBMClassifier,
            MyTunedLGBMClassifier,
            MyTunedESLGBMClassifier,
            MyEnsembledLGBMClassifier,
            MyEnsembledESLGBMClassifier,
            MyTabPFNClassifier,
            MyTunedTabPFNClassifier,
            MyEnsembledTabPFNClassifier,
            FamilyEnsembleEstimator,
            TabularPredictor
        )
    ):
        raise TypeError("The deserialized object isn't an estimator.")
    


def check_estimator_is_fitted(obj) -> None:
    if isinstance(obj, FamilyEnsembleEstimator):
        if not hasattr(obj, "ensembles_"):
            raise ValueError("The estimator is not fitted.")
    elif isinstance(obj, TabularPredictor):
        if not obj.is_fit:
            raise ValueError("The estimator is not fitted.")
    else:
        if not hasattr(obj, "estimator_"):
            raise ValueError("The estimator is not fitted.")



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("-f", "--file-estimator", required=True,
                   help="Pickle file of the fitted estimator to use in inference.")
    
    p.add_argument("-i", "--input-data", required=True, 
                   help="Path of the file/folder containing the data for inference.")
    
    p.add_argument("-o", "--output-dir", required=True, help="Output folder path.")
    
    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"],
                   help=h(
                    """Defines the input data format. One of 'xy' and 'df'.
                    -xy: A folder containing `X.txt` and `y.txt` named files.
                    -df: A file containing both X and y data.
                    In both cases the program DEMANDS tab-separated text files."""))
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'.")
    
    p.add_argument("-x", "--x-uniform", action="store_true",
                   help="Uniform the input data feature space to the one seen by the estimator at fit level.")

    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    p.add_argument("--disable-additional-txt-output", action="store_true", 
                   help=h("""
                    Disable the generation of the txt file with the predicted estimator probabilities.
                   In this case the predictions are only available in the main output in an encoded format,
                   which requires the package python API for decoding."""))
    
    return p.parse_args(args)




def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_target_feature(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    logger = create_logger(sys.stdout)

    # deserialize estimator
    with open(pars["file_estimator"], "rb") as f:
        estimator: Estimator | FamilyEnsembleEstimator | TabularPredictor = pickle.load(f)

    check_is_estimator_object(estimator)
    check_estimator_is_fitted(estimator)
    logger.debug("Estimator deserialized!")

    dl = DataLoader()

    # load data
    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y = dl.X, dl.y
    predict_dataset_name = dl.generic_dataset_name
    logger.debug("Data loaded in memory!")

    # get fit info from the estimator
    fit_dataset_name = estimator._info_predict_program_["fit_dataset_name"]
    le = estimator._info_predict_program_["le"]
    classes = estimator._info_predict_program_["classes"]
    classes_counts = estimator._info_predict_program_["classes_counts"]
    fit_features = estimator._info_predict_program_["fit_features"]

    # encode y
    y_enc = le.transform(y)

    if isinstance(estimator, (FamilyEnsembleEstimator, TabularPredictor)):
        fit_preprocessing_dict = {}
    else:
        fit_preprocessing_dict = estimator.collect_fit_preprocessing_info()
    
    # uniform feature space when requested
    if pars["x_uniform"]:
        if not np.isin(X.columns.to_numpy(), fit_features).any():
            raise ValueError(
                "Test feature space has no feature in common with the training space."
            )
        X = X.reindex(columns=fit_features, fill_value=0.0)
        logger.debug("Test feature space uniformed to training space.")
    
    # get predictions
    pred_proba = estimator.predict_proba(X, as_pandas=False) \
        if isinstance(estimator, TabularPredictor) \
        else estimator.predict_proba(X)

    pdf = PredictionDataframe()

    pdf.build_from_data(
        dataset=fit_dataset_name,
        y_test=y_enc,
        pred_proba=pred_proba,
        classes=classes,
        classes_counts=classes_counts,
        save_path=None,
        predict_dataset=predict_dataset_name,
        preprocessing=getattr(estimator, "preprocessing", None),
        **fit_preprocessing_dict
    )
    
    pdf.compute_metrics(multiclass="average", average_strategy="macro")
    filename = f"pred_dataframe__{fit_dataset_name}__{predict_dataset_name}.txt"
    pdf.to_csv(pars["output_dir"] / filename, sep="\t", index=False)
    
    if not pars["disable_additional_txt_output"]:
        np.savetxt(
            pars["output_dir"] / "predicted_probabilities.txt",
            pred_proba,
            delimiter="\t"
        )
    
    logger.debug(f"Output created at: {pars["output_dir"]}.")



if __name__ == "__main__":
    main()