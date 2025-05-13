import os
import sys
import json
import pandas as pd
from functools import partial
from sklearn.pipeline import Pipeline
from tabutils.prediction import PredictionDataframe
from tabutils.percentage import filter_percentage, get_filtering_thresh
from metatab.params import parse_args, check_args, adjust_args
from metatab.constants import PRED_DATAFRAME_ADDITIONAL_COLUMNS
from metatab.load import DataLoader
from metatab.log import create_logger, log_iteration
from metatab.run_model import pick_splitter, pick_classifier, create_classifier_pipeline, get_repetition_fold
from metatab.save import create_dict_hpo, create_dict_results, populate_dict_result_, populate_dict_hpo_, create_configuration_dict, get_classifier_filename, save_classifier



def main():

    # collect args
    pars = vars(parse_args(sys.argv[1:]))
    check_args(pars)
    pars = adjust_args(pars)


    # set variables
    data_loader = DataLoader()
    stdout_logger = create_logger(sys.stdout)
    do_without_preprocessing = True if "no" in pars["preprocessing"] else False
    do_filtering = True if "filter" in pars["preprocessing"] else False
    do_pca = True if "pca" in pars["preprocessing"] else False
    name_dataset = pars["input_path"].stem
    name_test_dataset = pd.NA if pars["test_dataset"] is None else pars["test_dataset"].stem


    # create output folder
    output_path = pars["output_path"]
    os.makedirs(output_path, exist_ok=True)
    if pars["save_models"]:
        os.makedirs(output_path / "models", exist_ok=True)


    # load data
    load_as = "train" if pars["test_dataset"] else "generic"
    
    data_loader.load_data(
        mode=pars["input_mode"], 
        path=pars["input_path"], 
        target_feature=pars["target_feature"], 
        load_as=load_as
    )

    if pars["test_dataset"]:
        data_loader.load_data(
            mode=pars["input_mode"], 
            path=pars["test_dataset"],
            target_feature=pars["target_feature"],
            load_as="test"
        )
        data_loader.X_test = data_loader.X_test.reindex(columns=data_loader.X_train.columns, fill_value=0.0)
        if data_loader.X_test.to_numpy().sum() == 0:
            raise ValueError("The training and testing datasets have no feature in common! Is impossible to proceed.")

    stdout_logger.debug("Data correctly loaded in memory\n")


    splitter = pick_splitter(pars)
    dict_results = create_dict_results()
    dict_hpo = create_dict_hpo(pars)
    

    # run the model
    for i, (idx_train, idx_test) in enumerate(splitter.split(data_loader.X, data_loader.y)):
        repetition, fold = get_repetition_fold(i, pars)
        log_iteration(pars, fold, repetition, stdout_logger)
        
        X_train, y_train, X_test, y_test = data_loader.return_train_test_xy_sets(idx_train, idx_test)

        partial_populate_dict_result_ = partial(
            populate_dict_result_,
            dict_results=dict_results,
            dataset=name_dataset,
            y_train=y_train,
            y_test=y_test,
            model=pars["model"],
            test_dataset=name_test_dataset,
            splitting_mode=pars["splitting_mode"],
            repetition=repetition,
            fold=fold
        )


        if do_without_preprocessing:
            clf = pick_classifier(pars)
            clf_piped = create_classifier_pipeline(clf, "no", pars)
            clf_piped.fit(X_train, y_train)
            pred_proba = clf_piped.predict_proba(X_test)
            partial_populate_dict_result_(pred_proba=pred_proba, preprocessing="no")
            populate_dict_hpo_(dict_hpo, clf_piped, pars["model"], pars["splitting_mode"], "no", repetition, fold)
            model_filename = get_classifier_filename(pars, repetition, fold)
            save_classifier(clf_piped, model_filename, pars["save_models"], "no")
            stdout_logger.debug("\t -Completed inference with no preprocessing")

        
        if do_filtering:
            number_initial_features = X_train.shape[1]
            filt_thresh = get_filtering_thresh(X_train, 499)

            if filt_thresh == 0 and do_without_preprocessing:
                stdout_logger.debug("\t -Warning: the training set has less than 500 features. \
                    The inference is skipped since it is equal to the 'no' preprocessing scenario.")
            elif filt_thresh is None and do_without_preprocessing:
                stdout_logger.debug("\t -Warning: the filtering procedure cannot reduce the number of features under 500. \
                    The inference is skipped since it's equal to the 'no' preprocessing scenario.")
            else:
                # here we must run the model indipendently of filtering
                X_train_filtered = X_train
                X_test_filtered = X_test
                number_filtered_features = 0

                if filt_thresh is None:
                    stdout_logger.debug("\t -Warning: the filtering procedure cannot reduce the number of features under 500. No filtering is done.")
                elif filt_thresh == 0:
                    stdout_logger.debug("\t -Warning: the training set has less than 500 features. No filtering is done.")
                else:
                    X_train_filtered = filter_percentage(X_train, filt_thresh)
                    X_test_filtered = X_test.reindex(columns=X_train_filtered.columns)
                    number_filtered_features = number_initial_features - X_train_filtered.shape[1]
                
                clf = pick_classifier(pars)
                clf_piped = create_classifier_pipeline(clf, "filter", pars)
                clf_piped.fit(X_train_filtered, y_train)
                pred_proba = clf_piped.predict_proba(X_test_filtered)

                model_filename = get_classifier_filename(pars, repetition, fold, "no")
                save_classifier(clf_piped, model_filename, pars["save_models"])

                populate_dict_hpo_(dict_hpo, clf_piped, pars["model"], pars["splitting_mode"], "filter", repetition, fold)

                partial_populate_dict_result_(
                    pred_proba=pred_proba, 
                    preprocessing="filter",
                    number_initial_features=number_initial_features,
                    number_filtered_features=number_filtered_features, 
                    filtering_threshold=filt_thresh
                )

                stdout_logger.debug("\t -Completed inference with feature filtering preprocessing")
        

        if do_pca:
            clf = pick_classifier(pars)
            clf_piped = create_classifier_pipeline(clf, "pca", pars)
            clf_piped.fit(X_train, y_train)
            clf_piped.predict_proba(X_test)
            pca = clf_piped.named_steps["pca"] if isinstance(clf_piped, Pipeline) else clf_piped.best_estimator_.named_steps["pca"]

            model_filename = get_classifier_filename(pars, repetition, fold, "pca")
            save_classifier(clf_piped, model_filename, pars["save_models"])
            
            populate_dict_hpo_(dict_hpo, clf_piped, pars["model"], pars["splitting_mode"], "pca", repetition, fold)

            partial_populate_dict_result_(
                pred_proba=pred_proba, 
                preprocessing="pca",
                number_pca_components=pca.n_components_
            )
            
            stdout_logger.debug("\t -Completed inference with PCA preprocessing \n")


    # compute performance metrics and save the results
    path_pred_dataframe = output_path / "pred_dataframe.txt"
    path_configuration_file = output_path / "configuration.json"
    path_hpo = output_path / "hpo.txt"

    df_pred = PredictionDataframe()

    df_pred.build(
        dataset=dict_results["dataset"], 
        y_train=dict_results["y_train"], 
        y_test=dict_results["y_test"], 
        pred_proba=dict_results["pred_proba"],
        save_path=output_path,
        **{key: dict_results[key] for key in PRED_DATAFRAME_ADDITIONAL_COLUMNS}
    )

    if not df_pred.has_recovered:
        df_pred.compute_metrics()
        df_pred.to_csv(path_pred_dataframe, sep="\t", index=False)

    if (pars["model"] == "rf" and pars["grid_search"] is not None) or pars["model"] == "ft_opt":
        pd.DataFrame(dict_hpo).to_csv(path_hpo, sep="\t", index=False)
        
    with open(path_configuration_file, "w") as f:
        conf_dict = create_configuration_dict(pars)
        json.dump(conf_dict, f, indent=4)

    stdout_logger.debug(f"Outputs created at {output_path}")



if __name__ == "__main__":
    main()