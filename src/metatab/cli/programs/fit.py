"""
Program to fit an ML estimator on a dataset.
The program seralizes the fitted model in a binary file via pickle.
"""

import sys
import pickle
import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.general import create_unique_column_name
from metatab.estimators.utils.general import check_meta_tuning_options
from metatab.estimators.utils.pick import pick_estimator_class
from metatab.estimators.estimators import Estimator
from metatab.ensemble.family import FamilyEnsembleEstimator
from metatab.ensemble.utils import BagCV
from metatab.preprocessing.density_selector import DensityFeatureSelector
from metatab.metalearning.load import query_surrogate_framework
from metatab.estimators.params.good_default import GOOD_DEFAULTS_MAP

from metatab.cli.helper import (
    adjust_io_paths_, 
    manage_output_path,
    check_early_stop_parameters,
    check_device,
    check_target_feature,
    create_logger,
    build_early_stop_configuration,
    build_tune_configuration,
    build_ensemble_configuration,
    get_ensemble_configuration,
    download_required_surrogate_models,
    create_json_configuration_file,
    add_predict_attrs_to_estimator
)

from metatab.cli.parser import (
    make_base_parser,
    make_fit_parser,
    make_tune_parser,
    make_extra_base_parser,
    make_ensemble_parser,
    make_family_ensemble_parser,
    make_autogluon_parser
)



def parse_args(args):
    p = ArgumentParser(formatter_class=RawTextHelpFormatter)
    # add subparser for estimator mode
    sub_estimator_mode = p.add_subparsers(required=True, title="Estimator Mode", description="valid subcommands")
    p_default = sub_estimator_mode.add_parser("default", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser()], formatter_class=RawTextHelpFormatter)
    p_good_default = sub_estimator_mode.add_parser("good_default", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser()], formatter_class=RawTextHelpFormatter)
    p_tune = sub_estimator_mode.add_parser("tune", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser(), make_tune_parser()], formatter_class=RawTextHelpFormatter)
    p_ensemble = sub_estimator_mode.add_parser("ensemble", parents=[make_base_parser(), make_extra_base_parser(), make_fit_parser(), make_ensemble_parser()], formatter_class=RawTextHelpFormatter)
    p_family_ensemble = sub_estimator_mode.add_parser("family-ensemble", parents=[make_base_parser(), make_fit_parser(), make_family_ensemble_parser()], formatter_class=RawTextHelpFormatter)
    p_autogluon = sub_estimator_mode.add_parser("autogluon", parents=[make_base_parser(), make_autogluon_parser()], formatter_class=RawTextHelpFormatter)
    p_default.set_defaults(estimator_mode="default")
    p_good_default.set_defaults(estimator_mode="good_default")
    p_tune.set_defaults(estimator_mode="tune")
    p_ensemble.set_defaults(estimator_mode="ensemble")
    p_family_ensemble.set_defaults(estimator_mode="family_ensemble")
    p_autogluon.set_defaults(estimator_mode="autogluon")
    return p.parse_args(args)



def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))

    check_target_feature(pars)

    # autogluon has no device parameter
    if pars.get("device", None) is not None:
        check_device(pars)

    if pars["estimator_mode"] not in ["family_ensemble", "autogluon"]:
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
        load_as="generic"
    )

    X, y = dl.X, dl.y
    fit_dataset_name = dl.generic_dataset_name
    logger.debug("Data loaded in memory!")

    # encode y
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y)) # to have Xy "type" uniformity
    

    if pars["estimator_mode"] == "family_ensemble":
        configuration = get_ensemble_configuration(pars["ensemble_configuration"])
        # this is to avoid the first download inside the fit call inflating times
        download_required_surrogate_models(configuration)

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

        estimator.fit(X, y_enc)
    
    elif pars["estimator_mode"] == "autogluon":
        y_enc.name = pars["target_feature"] if pars["input_mode"] == "df" else create_unique_column_name(X, "_target_")
        
        density_selector = DensityFeatureSelector(
            n_target_cols=pars["n_columns_density_filter"],
            strategy="exact",
            on_empty="error"
        ).set_output(transform="pandas")
        
        X = density_selector.fit_transform(X)
        data = pd.concat([X, y_enc], axis=1)
        
        estimator = TabularPredictor(
            label=y_enc.name,
            eval_metric=pars["eval_metric"],
            path=str(pars["output_dir"]),
            verbosity=0
        )

        estimator.fit(
            train_data=data,
            presets=pars["preset"],
            time_limit=pars["time_limit"],
            num_cpus=pars["nthreads"],
            num_gpus=pars["ngpus"],
            auto_stack=True            
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
            preprocessing=pars["preprocessing"],
            seed=pars["seed"],
            n_threads=pars["nthreads"],
            device=pars["device"],
            early_stop_configuration=early_stop_conf,
            tune_configuration=tune_conf,
            ensemble_configuration=ens_conf
        )

        # set good defaults when requested
        if pars["estimator_mode"] == "good_default":
            estimator.fixed_params = GOOD_DEFAULTS_MAP[pars["estimator"]]

        estimator.fit(X, y_enc)


    logger.debug("Estimator fitted on training data.")
    
    # we set attributes requested by the predict program
    add_predict_attrs_to_estimator(
        estimator=estimator,
        label_encoder=le,
        X_train=X,
        y_train=y_enc,
        fit_dataset_name=fit_dataset_name
    )

    out_filepath = pars["output_dir"] / "estimator.pkl"
    
    if pars["estimator_mode"] != "autogluon":
        estimator.save(out_filepath, check_is_fitted=True)
    else:
        with open(out_filepath, "wb") as f:
            pickle.dump(estimator, f)

    logger.debug(f"Estimator serialized in '{out_filepath}'.")




if __name__ == "__main__":
    main()