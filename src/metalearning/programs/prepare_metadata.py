"""Program to adapt and prepare search data as meta-data.

In addition to the seach data, the program needs external info, in detail:
- the dataset on which the search data has been generated
- the preprocessing option used for the search
"""

import sys
import argparse
import logging
import pandas as pd
from metatab_utils.data_loader import DataLoader
from metalearning.metafeatures import extract_metafeatures

from metatab_utils.helper_programs import (
    create_logger,
    adjust_paths_,
    check_target_feature,
    manage_output_path
)




def parse_args(args):
    p = argparse.ArgumentParser()

    p.add_argument("-i", "--input-file", required=True, help="Input search data filepath.")

    p.add_argument("-o", "--output-file", required=True, help="Output filepath.")

    p.add_argument("-d", "--data-file", required=True, 
                   help="""Filepath of the data on which the search data has been generated. Needed to obtain metafeatures.""")
    
    p.add_argument("-m", "--input-mode", required=True, choices=["xy", "df"], 
                   help="Defines the data input format. One between 'xy' and 'df'.")
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")

    p.add_argument("-p", "--preprocessing", required=True, choices=["base", "density_filter", "pca"],
                   help="One must specify the preprocessing option used to generate the search data.")
    
    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")

    return p.parse_args(args)



def log_program_setting(pars: dict, logger: logging.Logger) -> None:
    logger.debug(
        (
            f"Launching the prepare-metadata program on '{pars["input_file"]}' as search data"
            f" and '{pars["data_file"]}' as data, with '{pars["preprocessing"]}' as preprocessing info.\n"

        )
    )



def aggregate_df_search(
    df_search: pd.DataFrame, 
    groupby_column: str = "search_iter",
    loss_column: str = "loss",
    remove_groupby_column: bool = False
) -> pd.DataFrame:
    '''
    Abstract the logic to aggregate the df search.
    Apply mean aggregation on the loss column and first aggregation on the others.
    Returns the aggragated dataframe.
    '''
    agg_dict = {}
    for col in df_search.columns:
        agg_func = "mean" if col == loss_column else "first"
        agg_dict[col] = agg_func

    del agg_dict[groupby_column]
    df_search_agg = df_search.groupby(groupby_column).agg(agg_dict).reset_index()

    if remove_groupby_column:
        del df_search_agg[groupby_column]

    return df_search_agg




def main():
    logger = create_logger(sys.stdout)
    pars = vars(parse_args(sys.argv[1:]))
    check_target_feature(pars)
    adjust_paths_(pars, "input_file", "output_file", "data_file")
    manage_output_path(pars, "output_file", False)
    
    log_program_setting(pars, logger)

    # the sep specification is uniformed accross our programs
    df_search = pd.read_csv(pars["input_file"], sep="\t")
    logger.debug("Search data loaded in memory!")

    dl = DataLoader()

    dl.load(
        mode=pars["input_mode"],
        path=pars["data_file"],
        target_feature=pars["target_feature"],
        load_as="generic"
    )

    X, y = dl.X, dl.y
    logger.debug("Data loaded in memory!")

    # remove the non useful columns
    del df_search["fold"]
    del df_search["repeat"]
    
    df_search_agg = aggregate_df_search(df_search, remove_groupby_column=True)
        
    # z-normalize the loss column
    loss_col = df_search_agg["loss"]
    df_search_agg["z_normalized_loss"] = (loss_col - loss_col.mean()) / loss_col.std()
    del df_search_agg["loss"]

    # add preprocessing column
    df_search_agg["preprocessing"] = pars["preprocessing"]
    
    # add metafeatures
    metafeatures = extract_metafeatures(X, y)
    for metafeature, value in metafeatures.items():
        df_search_agg[metafeature] = value

    # save
    df_search_agg.to_csv(pars["output_file"], sep="\t", index=False)
    logger.debug(f"Output saved at {pars["output_file"]}")




if __name__ == "__main__":
    main()