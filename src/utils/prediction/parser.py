import re
import numpy as np
import pandas as pd
from utils.prediction.constants import PERFORMANCE_METRICS



def parse_pred_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Parse back from str to numpy arrays specific columns of the PredictionDataFrame.
    Note that the function wants in input the underlying dataframe.
    Return the parsed underlying dataframe.
    '''
    return df.apply(_parse_row, axis=1)


def _parse_row(row: pd.Series):
    '''Parse a single row of the PredictionDataFrame'''
    # pred_proba is processed on its own
    must_labels_to_parse = ["classes", "classes_counts", "test_labels", "pred_labels"]
    metrics_labels_to_parse = PERFORMANCE_METRICS
    all_labels = must_labels_to_parse + metrics_labels_to_parse

    for label in all_labels:
        dtype = np.float64 if label in metrics_labels_to_parse else np.int64
        if label in row.index:
            value = row[label]
            # metrics can be NA, float or np arrays
            row[label] = value \
                if pd.isna(value) or isinstance(value, float) \
                else np.fromstring(re.sub(r"[\[\]\n]", "", value), sep=" ", dtype=dtype) 

    row["pred_proba"] = row["pred_proba"] \
        if pd.isna(row["pred_proba"]) \
        else np.reshape(
            np.fromstring(re.sub(r"[\[\]\n]", "", row["pred_proba"]), sep=" ", dtype=np.float64), 
            shape=(-1, row["classes"].size)
        ) 
    
    return row
