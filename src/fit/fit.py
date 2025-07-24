import sys
from estimators.types import Estimator
from metatab_utils.data_loader import DataLoader
from metatab_utils.general import create_logger, check_y_is_integer_encoded
from fit.fit_helper import pick_estimator_class
from fit.params import parse_args

from metatab_utils.helper_params import (
    adjust_io_paths_, 
    manage_output_path,
    check_fit_resample_args,
    adjust_tune_configuration_arg_
)


def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_fit_resample_args(pars)

    adjust_io_paths_(pars, "input_data", "output_path")
    manage_output_path(pars, "output_path", False)
    adjust_tune_configuration_arg_(pars)
    
    logger = create_logger(sys.stdout)
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="train",
        skip=["X_test", "y_test"]
    )

    X_train, y_train = dl.X_train, dl.y_train
    check_y_is_integer_encoded(y_train)
    logger.debug("Data loaded in memory!")

    # here we consider all 3 load methods to retrieve the dataset_name
    fit_dataset_name = dl.train_dataset_name if dl.train_dataset_name else dl.generic_dataset_name
    estimator_class = pick_estimator_class(pars)
    
    estimator: Estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"],
        n_cores=pars["ncores"],
        tune_configuration=pars["tune_configuration"]
    )
    
    ## TODO: here we must implement an universal fit adapter
    ## when estimators with a different fit signature are implemented
    estimator.fit(X_train, y_train)
    logger.debug("Estimator fitted on input data.")
    
    # we set y_train and fit_dataset_name since are requested by the predict program 
    estimator._y_train_ = y_train
    estimator._fit_dataset_name_ = fit_dataset_name

    estimator.save(pars["output_path"])
    logger.debug(f"Estimator serialized in '{pars["output_path"]}'.")



if __name__ == "__main__":
    main()
    
