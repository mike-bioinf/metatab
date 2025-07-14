import sys
from utils.data_loader import DataLoader
from utils.logging import create_logger
from utils.helper_params import adjust_io_paths_
from fit.params import parse_args, check_args, manage_output_path
from fit.main_helper import pick_estimator



def main():
    pars = vars(parse_args(sys.argv[1:]))
    check_args(pars)

    adjust_io_paths_(pars)
    manage_output_path(pars)
    
    logger = create_logger(sys.stdout)
    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_path"],
        target_feature=pars["target_feature"],
        load_as="train",
        save_missing=["X_test", "y_test"]
    )

    X_train, y_train = dl.X_train, dl.y_train
    logger.debug("Data loaded in memory!")

    estimator_class = pick_estimator(pars)
    
    estimator = estimator_class(
        preprocessing=pars["preprocessing"],
        seed=pars["seed"]
    )
    
    estimator.fit(X_train, y_train)
    logger.debug("Estimator fitted on input data.")

    estimator.save(pars["output_path"])
    logger.debug(f"Estimator serialized in '{pars["output_path"]}'.")



if __name__ == "__main__":
    main()
    
