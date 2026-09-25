"""Script used to generate metadata

We generate the metadata from the search data, metafeatures and (eventually) preprocessing info.
The three components are concatenated in the right order since the resulting feature order is relevant 
in the fitting and predict procedures of the the surrogate framework. We want the hps-metafeatures-preprocessing order.
"""

import sys
import argparse
import warnings
import pandas as pd
from pathlib import Path
from metatab.metalearning.metafeatures import CustomMFE



def parse_args(args):
    p = argparse.ArgumentParser()
    
    p.add_argument("--profiles-folder", required=True, help="Path to the folder with the microbial profiles.")
    
    p.add_argument("--search-data-folder", required=True, 
                   help="""Path to the folder with the search data for every estimator.
                   The expected folder structure is 'estimator_folder' with txt files of search data nominated after the datasets.""")
    
    p.add_argument("--target-feature", default="Group", help="Name of the target feature column.")

    p.add_argument("--preprocessing", 
                   help="""
                   If specified, add this string as the last column in the metadata.
                   If unspecified, search for the preprocessing column and set it as last column in metadata.
                   """)
    
    p.add_argument("--output-folder", required=True, help="Path of the output folder")

    return p.parse_args(args)



def aggregate_df_search(df_search: pd.DataFrame) -> pd.DataFrame:
    '''
    Abstract the logic to aggregate the df search.
    Apply mean aggregation on the loss column and first aggregation on the others.
    Returns the aggregated dataframe.
    '''
    agg_dict = {}
    for col in df_search.columns:
        agg_func = "mean" if col == "loss" else "first"
        agg_dict[col] = agg_func

    del agg_dict["search_iter"]
    df_search_agg = df_search.groupby("search_iter").agg(agg_dict).reset_index()
    del df_search_agg["search_iter"]
    return df_search_agg



def main():
    pars=vars(parse_args(sys.argv[1:]))
    profiles_folder = Path(pars["profiles_folder"])
    search_data_folder = Path(pars["search_data_folder"])
    output_folder=Path(pars["output_folder"])
    cmfe = CustomMFE()

    for path_dataset in profiles_folder.iterdir():
        name_dataset_file = path_dataset.name
        
        data = pd.read_table(path_dataset, sep="\t")
        X = data.drop(columns=pars["target_feature"])
        y = data[pars["target_feature"]]

        metafeatures, _ = cmfe.fit(X, y).extract()

        # we must add the metafeatures to the corresponding search data for every estimator
        for estimator in [
            "random_forest", "extra_trees", "lgbm", "es_lgbm", "xgb", "es_xgb", 
            "catboost", "es_catboost", "tabpfn", "tabm", "realmlp"
        ]:
            # create output folder
            path_estimator_out = output_folder / estimator
            path_estimator_out.mkdir(parents=True, exist_ok=True)
            
            for psde in (search_data_folder / estimator).iterdir():
                # here we skip when the search data and metafeatures are computed on different datasets
                if psde.name != name_dataset_file:
                    continue
                
                search_data = pd.read_table(psde, sep="\t")
                
                # remove the non useful columns
                del search_data["fold"]
                del search_data["repeat"]
                
                # we aggregate the search_data losses by point
                search_data_agg = aggregate_df_search(search_data)
                    
                # z-normalize the loss column
                loss_col = search_data_agg["loss"]
                search_data_agg["z_normalized_loss"] = (loss_col - loss_col.mean()) / loss_col.std()
                del search_data_agg["loss"]

                # we create a copy since the original df is not optimized in memory due to assign
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning)
                    search_data_agg = search_data_agg.assign(**metafeatures).copy()

                if pars["preprocessing"] is None:
                    search_data_agg = search_data_agg[
                        [col for col in search_data_agg.columns if col != "preprocessing"] + ["preprocessing"]
                    ]
                else:
                    search_data_agg["preprocessing"] = pars["preprocessing"]

                
                # save
                path_out = path_estimator_out / name_dataset_file
                search_data_agg.to_csv(path_out, sep="\t", index=False)