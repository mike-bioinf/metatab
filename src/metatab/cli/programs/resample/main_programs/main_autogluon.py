import os
import sys
import pickle
import shutil
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.prediction import PredictionDataframe
from metatab.metatab_utils.general import create_unique_column_name

from metatab.cli.programs.resample.helper import (
    pick_splitter,
    get_repetition_fold,
    log_iteration,
    populate_dict_lists_,
    get_resample_iteration_signature
)

from metatab.cli.helper import (
    create_logger,
    create_json_configuration_file,
    check_target_feature,
    check_holdout_train_size,
    adjust_io_paths_,
    manage_output_path,
    add_predict_attrs_to_estimator
)



def main_autogluon(pars: dict):
    logger = create_logger(sys.stdout)

    check_target_feature(pars)
    check_holdout_train_size(pars)

    adjust_io_paths_(pars, "input_data", "output_dir")
    manage_output_path(pars, "output_dir", True)

    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["input_data"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y = dl.X, dl.y
    name_dataset = dl.generic_dataset_name
    logger.debug(f"\nLaunching autogluon '{pars["preset"]}' on {name_dataset}!")

    # y encoding
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))
    y.name = pars["target_feature"] if pars["input_mode"] == "df" else create_unique_column_name(X, "_target_")

    # initialize outputs
    output_dir = pars["output_dir"]
    results_filepath = output_dir / "pred_dataframe.txt"
    configuration_filepath = output_dir / "configuration.json"
    path_estimators = pars["output_dir"] / "estimators" 
    create_json_configuration_file(pars, configuration_filepath)
    dict_results = defaultdict(list)
    df_pred_results = PredictionDataframe()

    if not pars["disable_additional_txt_output"]: 
        txt_folder = output_dir / "additional_txt_info"
        txt_folder.mkdir(exist_ok=True)

    splitter = pick_splitter(pars)
    

    # run resampling
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repetition, fold = get_repetition_fold(i, pars)
        iter_signature = get_resample_iteration_signature(repetition, fold)
        log_iteration(pars, fold, repetition, logger)
        
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        train_data = pd.concat([X_train, y_train], axis=1)
        path_iteration = str(path_estimators / f"estimator_{iter_signature}")

        autogluon_predictor = TabularPredictor(
            label=y.name,
            eval_metric=pars["eval_metric"],
            path=path_iteration,
            verbosity=0
        )

        t = time()
        autogluon_predictor.fit(
            train_data=train_data,
            presets=pars["preset"],
            time_limit=pars["time_limit"],
            num_cpus=pars["nthreads"],
            num_gpus=pars["ngpus"],
            auto_stack=True
        )
        fit_time = time() - t

        logger.debug("\t-Estimator fitted on input data.")
        logger.debug(f"\t-Fit time in minutes: {round(fit_time/60, 2)}")
        
        t = time()
        pred_proba = autogluon_predictor.predict_proba(X_test, as_pandas=False)
        predict_time = time() - t
        logger.debug(f"\t-Inference time in minutes: {round(predict_time/60, 2)}\n")

        # store and/or save the iteration info
        iter_results = {
            "dataset": name_dataset,
            "predict_dataset": name_dataset,
            "estimator": "autogluon",
            "estimator_mode": "autogluon",
            "n_threads": pars["nthreads"],
            "n_gpus": pars["ngpus"],
            "splitting_mode": pars["splitting_mode"],
            "repetition": repetition,
            "fold": fold,
            "classes": le.classes_,
            "classes_counts": np.unique(y_train.to_numpy(), return_counts=True)[1],
            "y_test": y_test,
            "pred_proba": pred_proba,
            "fit_time": fit_time,
            "predict_time": predict_time
        }

        populate_dict_lists_(dict_results, **iter_results)

        if not pars["save_estimators"]:
            shutil.rmtree(path_iteration)
        else:
            add_predict_attrs_to_estimator(autogluon_predictor, le, X_train, y_train, name_dataset)
            with open(os.path.join(path_iteration, "estimator.pkl"), "wb") as f:
                pickle.dump(autogluon_predictor, f)

        if not pars["disable_additional_txt_output"]:
            txt_folder_iter = txt_folder / f"iter_{iter_signature}"
            txt_folder_iter.mkdir(exist_ok=True)
            np.savetxt(txt_folder_iter / "predicted_probabilities.txt", pred_proba, delimiter="\t")
            np.savetxt(txt_folder_iter / "y_true.txt", y_test, fmt="%.1i", delimiter="\t")


    # remove the estimators folder
    if not pars["save_estimators"]:
        shutil.rmtree(path_estimators)

    if not pars["disable_additional_txt_output"]:
        np.savetxt(txt_folder / "classes.txt", le.classes_, fmt="%.1000s", delimiter="\t")
    
    df_pred_results.build_from_data(**dict_results, save_path=output_dir)

    if not df_pred_results.has_recovered:
        df_pred_results.compute_metrics(multiclass="average", average_strategy="macro")
        df_pred_results.to_csv(results_filepath, sep="\t", index=False)

    logger.debug(f"Outputs created at {output_dir}")