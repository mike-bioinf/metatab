"""
Set of functions to check the correctness of the results obtained with our programs.
"""
import pandas as pd
from _paper.analysis.utils import check_presence_cols



def check_number_resample_cv_rounds(
    df: pd.DataFrame, 
    group_columns: str | list[str],
    expected_n_rounds: int
) -> None:
    '''
    Checks the number of cv rounds for every group identified by the group_columns.

    Parameters:
        df (pd.DataFrame): 
            Dataframe on which perform the check.
        
        group_columns (str | list[str]): 
            Columns to which group the dataframe.
            The check will be done on the single groups.

        expected_n_rounds (int):
            Expected number of rounds, i.e. number of rows of the groups.
    '''
    check_presence_cols(df, group_columns)
    for group, df_group in df.groupby(group_columns):
        group_n_rows = df_group.shape[0]
        if group_n_rows != expected_n_rounds:
            raise ValueError(
                f"Expected {expected_n_rounds} but found {group_n_rows} for the combination {group}"
            )
        


def check_nan_in_resample_hpo_dataframe(df_hpo: pd.DataFrame, regex_loss_columns: str = "loss*") -> None:
    '''
    Check on the presence of failed optimization iteration in the tuning process.
    The function is intented to work on the "hpo" dataframe, i.e. the dataframe 
    created by the resample program when tunining is involved.
    
    Parameters:
        df_hpo (pd.DataFrame):
            "hpo" dataframe, i.e. the one created by the resmaple program when a tune procedure is involved.
        
        regex_loss_columns (str):
            String to identify the loss columns, i.e the columns on which the check is done.
    '''
    df_losses = df_hpo.filter(regex=regex_loss_columns, axis=1)

    if df_losses.empty:
        raise ValueError(
            f"Zero columns are selected using the '{regex_loss_columns}' regular string."
        )
    
    if df_losses.isna().any().any():
        raise ValueError("Found nan values in the loss columns")