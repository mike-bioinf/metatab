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
    
    p.add_argument("--n-opt-iter", type=int, default=1500, help="The number of iterations in the search data to keep.")
    p.add_argument("--output-folder", required=True, help="Path of the output folder")
    return p.parse_args(args)



def prepare_search_data(df: pd.DataFrame, n_iter_keep: int) -> pd.DataFrame:
    '''
    Prepare the search data by:
    1. Selecting only the desired number of iterations starting from the top.
    2. Aggregating the iteration cv losses.
    3. Z-normalizing losses.
    '''
    # 1. select
    df = df.drop(columns=["fold", "repeat"])
    df = df.loc[df["search_iter"] < n_iter_keep, :]

    # 2. aggregate
    agg_dict = {}
    for col in df.columns:
        agg_func = "mean" if col == "loss" else "first"
        agg_dict[col] = agg_func

    del agg_dict["search_iter"]
    df_agg = df.groupby("search_iter").agg(agg_dict).reset_index()
    del df_agg["search_iter"]

    # 3. z-normalize loss
    loss_col = df_agg["loss"]
    df_agg["z_normalized_loss"] = (loss_col - loss_col.mean()) / loss_col.std()
    del df_agg["loss"]

    return df_agg



def prepare_metadata(search_data: pd.DataFrame, metafeatures: dict, pars: dict):
    '''
    Prepare metadata from search data and metafeatures. In particular does 2 things:
    1. Add metafeatures to search data.
    2. Manage the preprocessing column.
    '''
    # we create a copy since the original df is not optimized in memory due to assign
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning)
        metadata = search_data.assign(**metafeatures).copy()

    if pars["preprocessing"] is None:
        metadata = metadata[
            [col for col in metadata.columns if col != "preprocessing"] + ["preprocessing"]
        ]
    else:
        metadata["preprocessing"] = pars["preprocessing"]

    return metadata




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
                if psde.name != name_dataset_file: continue
                search_data = pd.read_table(psde, sep="\t")
                search_data = prepare_search_data(search_data, pars["n_opt_iter"])
                metadata = prepare_metadata(search_data, metafeatures, pars)
                metadata.to_csv(path_estimator_out / name_dataset_file, sep="\t", index=False)




if __name__ == "__main__":
    main()