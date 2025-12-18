"""
Program to fit an ML estimator on a dataset.
The program seralizes the fitted model in a binary file via pickle.
"""

import sys
import argparse
from metatab.metatab_utils.data_loader import DataLoader
from metatab.estimators.estimators import Estimator
from metatab.estimators.utils.pick import pick_estimator_class
from metatab.estimators.utils.general import check_y_is_integer_encoded, check_meta_tuning_options
from metatab.metalearning.load import query_surrogate_framework
from metatab.ensemble.family import FamilyEnsembleEstimator
from metatab.ensemble.utils import BagCV

from metatab.cli.helper import (
    adjust_io_paths_, 
    manage_output_path,
    check_early_stop_parameters,
    check_target_feature,
    create_logger,
    build_early_stop_configuration,
    build_tune_configuration,
    build_ensemble_configuration,
    resolve_preprocessing_info,
    get_ensemble_configuration,
    downaload_required_surrogate_models,
    create_json_configuration_file
)

from metatab.cli.parser import (
    make_base_parser,
    make_fit_parser,
    make_tune_parser,
    make_extra_base_parser,
    make_ensemble_parser,
    make_family_ensemble_parser
)



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sub_estimator_mode = p.add_subparsers(required=True, title="Estimator Mode", description="valid subcommands")
    p_default = sub_estimator_mode.add_parser("default", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser()])
    p_tune = sub_estimator_mode.add_parser("tune", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser(), make_tune_parser()])
    p_ensemble = sub_estimator_mode.add_parser("ensemble", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser(), make_ensemble_parser()])
    p_family_ensemble = sub_estimator_mode.add_parser("family-ensemble", parents=[make_base_parser(), make_fit_parser(), make_family_ensemble_parser()])
    p_default.set_defaults(estimator_mode="default")
    p_tune.set_defaults(estimator_mode="tune")
    p_ensemble.set_defaults(estimator_mode="ensemble")
    p_family_ensemble.set_defaults(estimator_mode="family_ensemble")
    return p.parse_args(args)



def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))

    check_target_feature(pars)

    if pars["estimator_mode"] != "family_ensemble":
        check_early_stop_parameters(pars)

    if (
        (pars["estimator_mode"] == "tune" and pars["tune_algo"] == "meta") or
        (pars["estimator_mode"] == "ensemble" and pars["ensemble_algo"] == "meta")
    ):
        space_attr = "ensemble_space" if pars["estimator_mode"] == "ensemble" else "tune_space"
        check_meta_tuning_options(
            pars["estimator"], 
            pars["preprocessing"], 
            pars[space_attr]
        )
        # this is to avoid the first download inside the fit call inflating times
        query_surrogate_framework(pars["estimator"])
    
    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)
    create_json_configuration_file(pars, filepath=pars["output_dir"] / "configuration.json")

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


    if pars["estimator_mode"] == "family_ensemble":
        configuration = get_ensemble_configuration(pars["ensemble_configuration"])
        # this is to avoid the first download inside the fit call inflating times
        downaload_required_surrogate_models(configuration)

        estimator = FamilyEnsembleEstimator(
            name=pars["ensemble_name"],
            configuration=configuration,
            save_path=pars["output_dir"] / "models",
            bag_cv=BagCV(pars["bag_cv_repats"], pars["bag_cv_folds"], pars["seed"]) if pars["use_bag_cv"] else None,
            feature_space_ratio=pars["feature_space_randomization"],
            seed=pars["seed"],
            time_limit=pars["ensemble_time_limit"],
            n_jobs=pars["nthreads"],
            log=50 #suppress logging
        )
    
    else:
        # build configuration objects
        ens_conf = None
        tune_conf = None
        early_stop_conf = build_early_stop_configuration(pars)

        if pars["estimator_mode"] == "ensemble":
            ens_conf = build_ensemble_configuration(pars)
        elif pars["estimator_mode"] == "tune":
            tune_conf = build_tune_configuration(pars)
        
        # get concrete estimator class
        estimator_class = pick_estimator_class(pars["estimator"], pars["estimator_mode"])
        
        estimator: Estimator = estimator_class(
            preprocessing=resolve_preprocessing_info(pars),
            seed=pars["seed"],
            n_threads=pars["nthreads"],
            early_stop_configuration=early_stop_conf,
            tune_configuration=tune_conf,
            ensemble_configuration=ens_conf
        )


    estimator.fit(X_train, y_train)
    logger.debug("Estimator fitted on training data.")
    
    # we set y_train and fit_dataset_name since are requested by the predict program 
    estimator._y_train_ = y_train
    estimator._fit_dataset_name_ = fit_dataset_name
    
    out_filepath = pars["output_dir"] / "estimator.pkl"
    estimator.save(out_filepath, check_is_fitted=True)
    logger.debug(f"Estimator serialized in '{out_filepath}'.")




if __name__ == "__main__":
    main()