'''
For all the functions defined in this module we are assuming in input a full numeric dataframe.
'''
import numpy as np
import pandas as pd
from preprocessing.utils import get_density_scores


HIGH_SPARSITY_THRESHOLD = 0.8
LOW_SPARSITY_THRESHOLD = 0.2


def compute_general_sparsity(df: pd.DataFrame) -> float:
    '''Returns the fraction of zero cells over their total number'''
    return (df == 0).sum().sum() / df.size
    

def compute_fraction_full_zero_columns(df: pd.DataFrame) -> float:
    '''Returns the fraction of full zero columns over their total number'''
    return (((df == 0).all(axis=0)).sum()) / df.shape[1]


def compute_fraction_full_dense_columns(df: pd.DataFrame) -> float:
    return (get_density_scores(df) == 1).sum() / df.shape[1]


def compute_fraction_high_sparsity_columns(df: pd.DataFrame) -> float:
    return ((1 - get_density_scores(df)) >= HIGH_SPARSITY_THRESHOLD).sum() / df.shape[1]


def compute_fraction_low_sparsity_columns(df: pd.DataFrame) -> float:
    return ((1 - get_density_scores(df)) <= LOW_SPARSITY_THRESHOLD).sum() / df.shape[1]


def compute_min_max_classes_ratio(y: pd.Series) -> float:
    # Does not consider "direction" of classes
    _, counts = np.unique(y.to_numpy(), return_counts=True)
    return np.min(counts) / np.max(counts)


def compute_stats_covariance(df: pd.DataFrame) -> tuple[float, float, float, float]:
    '''
    Computes the mean and std of the off diagonal elements of the covariance matrix,
    and of the diagonal elements (feature variances).
    Returns covs_mean, covs_std, vars_mean, vars_std in this order.
    '''
    cov_matrix = df.cov().to_numpy() 
    upper_triangle_indices = np.triu_indices_from(cov_matrix, k=1)
    cov_upper_triangle = cov_matrix[upper_triangle_indices]
    diagonal_indices = np.diag_indices_from(cov_matrix)
    diagonal = cov_matrix[diagonal_indices]
    return (
        cov_upper_triangle.mean(), 
        cov_upper_triangle.std(ddof=1), 
        diagonal.mean(),
        diagonal.std(ddof=1)
    )




def extract_metafeatures(X: pd.DataFrame, y: pd.Series) -> dict[str, int|float]:
    mean_cov_off_diagonal, std_cov_off_diagonal, mean_cov_diagonal, std_cov_diagonal = compute_stats_covariance(X)
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "sparsity": compute_general_sparsity(X),
        "fraction_full_zero_columns": compute_fraction_full_zero_columns(X),
        "fraction_high_sparsity_columns": compute_fraction_high_sparsity_columns(X),
        "fraction_low_sparsity_columns": compute_fraction_low_sparsity_columns(X),
        "fraction_full_dense_columns": compute_fraction_full_dense_columns(X),
        "min_max_classes_ratio": compute_min_max_classes_ratio(y),
        "mean_cov_off_diagonal": mean_cov_off_diagonal,
        "std_cov_off_diagonal": std_cov_off_diagonal,
        "mean_cov_diagonal": mean_cov_diagonal,
        "std_cov_diagonal": std_cov_diagonal
    }
