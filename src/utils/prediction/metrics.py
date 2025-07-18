import pandas as pd
import numpy as np
from typing import Literal

from sklearn.metrics import (
    average_precision_score,
    accuracy_score, 
    roc_auc_score, 
    precision_recall_fscore_support
)



def compute_metrics(
    row: pd.Series,
    multiclass: Literal["ovr", "average"],
    average_strategy: Literal["micro", "macro", "weighted"]
) -> pd.Series:
    '''
    Compute metrics supporting all classification scenarios.

    The function automatically distinguishes between binary and multiclass cases.
    Multiclass cases are further divided in straight one vs rest and averaged cases.

    Parameters:
        row (pd.Series): Prediction dataframe row.
        
        multiclass (Literal["ovr", "average"]): 
            Whether to compute and remain statistics in one vs rest format,
            or apply an averaging strategy on them/data (them/data because
            some strategy like "micro" does not pass by this ovr metrics).
            Note this parameter is ignored in binary cases.
        
        average_strategy (Literal["micro", "macro", "weighted"]):
            Average strategy to use. Is ignored if multiclass is not "average".
    
    Returns:
        pd.Series: A series of metrics.
    '''
    true_labels, pred_labels, pred_proba, classification_setting = (
        row["test_labels"], 
        row["pred_labels"], 
        row["pred_proba"], 
        row["classification_setting"]
    )

    if not isinstance(pred_proba, np.ndarray) and pd.isna(pred_proba):
        return pd.Series({
            "recall": np.nan, 
            "precision": np.nan, 
            "f1": np.nan, 
            "accuracy": np.nan, 
            "auc": np.nan
        })

    if classification_setting == "binary":
        pred_proba = pred_proba[:, 1]
        average_prf = "binary"
        average = None
        multi_class = "raise"
    elif multiclass == "ovr":
        average_prf = None
        average = None
        multi_class = "ovr"
    else:
        average_prf = average_strategy
        average = average_strategy
        multi_class = "ovr"

    precision, recall, f1, _  = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average=average_prf,
        zero_division=np.nan
    )

    accuracy = accuracy_score(true_labels, pred_labels)
    
    auc = roc_auc_score(
        true_labels, 
        pred_proba,
        average=average, 
        multi_class=multi_class
    )

    ap = average_precision_score(
        true_labels, 
        pred_proba,
        average=average
    )
    
    return pd.Series({
        "recall": recall, 
        "precision": precision, 
        "f1": f1, 
        "accuracy": accuracy,
        "ap": ap,
        "auc": auc
    })

