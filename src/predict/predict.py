import sys
import pickle
import numpy as np
from metatab_utils.data_loader import DataLoader
from metatab_utils.helper_programs import adjust_io_paths_, manage_output_path
from metatab_utils.general import check_y_is_integer_encoded, create_logger
from metatab_utils.prediction import PredictionDataframe
from predict.params import parse_args, check_args
from estimators import Estimator

from predict.predict_helper import (
    check_type_deserialized_object,
    check_estimator_is_fitted
)



def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_args(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    logger = create_logger(sys.stdout)

    # deserialize estimator
    with open(pars["file_estimator"], "rb") as f:
        estimator: Estimator = pickle.load(f)

    check_type_deserialized_object(estimator)
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

    X_test, y_test = dl.X_test, dl.y_test
    check_y_is_integer_encoded(y_test, is_predict_scenario=True)
    logger.debug("Data loaded in memory!")

    # here we consider all 3 load methods to retrieve the predict dataset name
    predict_dataset_name = dl.test_dataset_name if dl.test_dataset_name else dl.generic_dataset_name
    fit_dataset_name = estimator._fit_dataset_name_
    fit_preprocessing_dict = estimator.collect_fit_preprocessing_info()

    # uniform feature space if requested
    if pars["x_uniform"]:
        fit_features = estimator.get_feature_names_in_()
        if not np.isin(X_test.columns.to_numpy(), fit_features).any():
            raise ValueError(
                "Test feature space has no feature in common with the training space."
            )
        X_test = X_test.reindex(columns=fit_features, fill_value=0.0)
        logger.debug("Test feature space uniformed to training space.")
    
    pred_proba = estimator.predict_proba(X_test)
    pdf = PredictionDataframe()

    pdf.build_from_data(
        dataset=fit_dataset_name,
        y_train=estimator._y_train_,
        y_test=y_test,
        pred_proba=pred_proba,
        save_path=None,
        predict_dataset=predict_dataset_name,
        preprocessing=estimator.preprocessing,
        **fit_preprocessing_dict
    )
    
    pdf.compute_metrics(multiclass="average", average_strategy="macro")

    filename = f"pred_df__{fit_dataset_name}__{predict_dataset_name}.txt"
    filepath = pars["output_dir"] / filename
    pdf.to_csv(filepath, sep="\t", index=False)
    logger.debug(f"Output created to {filepath}.")



if __name__ == "__main__":
    main()