import sys
from utils.data_loader import DataLoader
from utils.logging import create_logger
from utils.helper_params import adjust_io_paths_, manage_output_path
from utils.general import check_y_is_integer_encoded
from estimators.types import Estimator
from fit.fit_helper import pick_estimator
from fit.params import parse_args, check_args



def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_args(pars)

    adjust_io_paths_(pars, "input_data", "output_path")
    manage_output_path(pars, "output_path", False)
    
    logger = create_logger(sys.stdout)
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="train",
        save_missing=["X_test", "y_test"]
    )

    X_train, y_train = dl.X_train, dl.y_train
    check_y_is_integer_encoded(y_train)
    logger.debug("Data loaded in memory!")

    # here we consider all 3 load methods to retrieve the dataset_name
    fit_dataset_name = dl.train_dataset_name if dl.train_dataset_name else dl.generic_dataset_name

    estimator_class = pick_estimator(pars)
    
    estimator: Estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"]
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
    
