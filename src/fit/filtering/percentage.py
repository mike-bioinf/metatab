import bisect
import pandas as pd
from warnings import warn



def filter_percentage(
    df: pd.DataFrame, 
    threshold_percent: int | float | None, 
    zero_na: bool = True
) -> pd.DataFrame:
    '''
    Filter numeric columns that do NOT satisfy a minimum 
    threshold (in percentage) of non-zero values.
    Columns are defined as numeric based on the `select_dtypes("number")` call.

    If is not possible to reach the desired number of columns or
    the df is empty, or does not contain "numeric" columns,
    then it is returned as it.

    Parameters:
        df (pd.DataFrame): Input pandas dataframe.
        
        threshold_percent (int | float | None): 
            Minimum percentage of non-zero values required to keep the column. 
            Must be specified as a percentage (e.g., 10). Must be a number in [0, 100]. 
            Fully dense columns (100% non-zero) are never removed.
            If None df is returned as is.
        
        zero_na (bool, optional): 
            Whether to treat NA, nan and None as 0s in percentage computation. 
            If False they are not considered in non-zero percentages computations. 

    Returns:
        pd.DataFrame: Filtered DataFrame.
    '''   
    if threshold_percent < 0 or threshold_percent > 100:
        raise ValueError("threshold_percent must be a number in [0, 100].")
    
    if threshold_percent is None or threshold_percent == 0:
        return df

    if df.empty or df.select_dtypes("number").empty:
        return df
    
    cols_percentages, _ = get_percentages(df, zero_na, 101)
    df_filtered = df.loc[:, cols_percentages >= threshold_percent]
    return df_filtered



def filter_zero_cols(df: pd.DataFrame, zero_na: bool = True) -> pd.DataFrame:
    '''
    Filter full 0 numeric columns.

    Parameters:
        df (pd.Dataframe): Input pandas dataframe.
        zero_na (bool, optional): 
            Whether to treat NA, nan and None as 0s in percentage computation. 
            If False they are not considered in computations.

    Returns:
        pd.Dataframe: The filtered pandas dataframe.
    '''
    if df.empty or df.select_dtypes("number").empty:
        return df
    
    percentage_series, _  = get_percentages(df, zero_na)
    df_filtered = df.loc[:, percentage_series != 0]
    return df_filtered



def get_filtering_thresh(
    df: pd.DataFrame, 
    target_ncols: int, 
    zero_na: bool = True, 
    quiet: bool = False
) -> float | None:
    '''
    Calculate the percentage of non-zero values required  
    to achieve the target number of columns with filtering.
    Returns None if is not possible to achieve the 
    target number in any means.

    Parameters:
        df (pd.Dataframe): Input pandas dataframe.
        target_ncols (int): Target number of columns.
        zero_na (bool, optional): 
            Whether to consider np.nan, NA and None as 0
            or not considering them in percentage computation.
        quiet (bool, optional): 
            Whether to suppress warnings. Defaults to False.

    Returns: 
        The percentage of non-zero values to filter by in order 
        to reach the target number of columns.
        Returns None if is not possible to achieve the target number in any means.
    '''
    ncols = df.shape[1]
    df_numeric = df.select_dtypes("number")
    ncols_numeric = df_numeric.shape[1]
    ncols_to_filter = ncols - target_ncols
    ncols_non_numeric = ncols - ncols_numeric

    # check if filtering is necessary
    if ncols_to_filter <= 0: 
        return 0
    
    # check if there are enough numeric columns to filter
    if ncols_to_filter > ncols_numeric:
        if not quiet: 
            warn(
                "The number of numeric columns to filter" +
                f" is not sufficient to reach {target_ncols}. None is returned."
            )
        return None
    
    # getting the numeric cols percentages
    perc_numeric_cols, _ = get_percentages(df_numeric, zero_na)
    
    # get the percentage of the "rightest" column to filter
    sorted_percs = sorted(perc_numeric_cols.to_list())
    target_perc = sorted_percs[ncols_to_filter - 1]
    
    # taking the next greater percentage to satisfy the target number
    index = bisect.bisect_right(sorted_percs, target_perc)
    
    # manage edge case of maximum target_perc
    filter_perc = sorted_percs[index] \
        if index < len(sorted_percs) \
        else _manage_rightend_perc(
            sorted_percs, 
            target_ncols, 
            ncols_non_numeric, 
            quiet
        )
    
    return filter_perc



def _manage_rightend_perc(
    sorted_percentages: list, 
    target_ncols: int,
    ncols_non_numeric: int,
    quiet: bool
) -> float|None:
    '''
    Internal of get_filtering_thresh that manages the scenario 
    in which the target filtering thresh is equal to the 
    maximum percentage of the dataframe.

    Parameters:
        sorted_percentages (list): 
            Sorted list of non-zero percentages of df columns.
        ncols_non_numeric (int): Number of non-numeric columns.
        target_ncols (int): Goal number of columns.
        quiet (bool): Whether to suppress warnings.

    Returns:
        float|None: The suggested filtering percentage. 
        Returns None if is not possible to reach the 
        target number of columns.
    '''
    target_perc = sorted_percentages[-1]
    ncols_rightend = len([perc for perc in sorted_percentages if perc == target_perc])
    ncols_evaluation = ncols_rightend + ncols_non_numeric
    
    if ncols_evaluation <= target_ncols:
        # keep the maximum percentage columns
        return target_perc
    elif target_perc < 100:
        # increase a tiny bit the percentage to filter all numeric columns
        to_max = 100 - target_perc
        return target_perc + to_max/100
    elif target_perc == 100:
        # we not filter 100% dense columns
        if not quiet: 
            warn(
                "There are not enough sparse columns to reach" +
                f" {target_ncols} columns. None is returned."
            )
        return None



def get_percentages(
    df: pd.DataFrame, 
    zero_na: bool, 
    default_percentage: int | float = 100
) -> tuple[pd.Series, list[str]]:
    '''
    Compute the percentages of non-zero values for all columns 
    of a dataframe. Allows to assign a default percentage value 
    to non-numeric columns.

    Parameters:
        df (pd.DataFrame): 
            Pandas dataframe.
        zero_na (bool): 
            Whether to consider nan and None values as 0,
            or not considering them.
        default_percentage (int | float, optional): 
            Percentage value assigned to non-numeric columns. 
            Defaults to 100.
    
    Returns:
        tuple: A tuple containing:
            - pd.Series: A Series object reporting for each column the non-zero percentage.
            - list[str]: A list of strings reporting the non-numeric column names.
    '''    
    if df.empty: 
        raise ValueError("df is empty.")

    numeric_cols = df.select_dtypes("number").columns
    non_numeric_cols = df.select_dtypes(exclude="number").columns.to_list()
    series_percentages = pd.Series(default_percentage, index=df.columns, dtype="float64")

    for col in numeric_cols:
        series_percentages.loc[col] = _get_percentage(df[col], zero_na)
    
    return series_percentages, non_numeric_cols



def _get_percentage(series: pd.Series, zero_na: bool) -> float:
    '''
    Compute the percentage of non-zero on a numeric series.
    Returns the percentage of non-zero value (0-100 range).
    '''
    if zero_na: 
        series = series.fillna(0.0)
    return (series != 0).mean(skipna=True) * 100
