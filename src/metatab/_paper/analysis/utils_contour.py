'''
Utilities for hp contourn plot visualization
'''

import numpy as np
import pandas as pd
from typing import Literal, Any
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.colors import Normalize, TwoSlopeNorm
from dataclasses import dataclass

VariableType = Literal["nominal", "ordinal", "continuous"]


def is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def apply_ordinal_order(
    categories: list,
    ordinal_order: list | None,
) -> list:
    """
    Return categories sorted by ordinal_order if provided,
    otherwise return them as-is (preserving original sequence).
    ordinal_order need not be exhaustive — any category not in it
    is appended at the end in their original relative order.
    """
    if ordinal_order is None: 
        return categories
    order_map = {v: i for i, v in enumerate(ordinal_order)}
    known = sorted([c for c in categories if c in order_map], key=lambda c: order_map[c])
    unknown = [c for c in categories if c not in order_map]
    return known + unknown


def build_norm(
    a: np.ndarray,
    mode: Literal["none", "min_max", "quantile", "centered_0"],
    qrange=(2, 98),
):
    a = a[np.isfinite(a)]
    if len(a) == 0 or mode == "none":
        return None
    if mode == "min_max":
        return Normalize(vmin=a.min(), vmax=a.max())
    if mode == "quantile":
        vmin, vmax = np.percentile(a, qrange)
        return Normalize(vmin=vmin, vmax=vmax)
    if mode == "centered_0":
        vmax = np.percentile(a, qrange[1])
        vmin = np.percentile(a, qrange[0])
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    raise ValueError(f"Unknown normalize_color: {mode}")


## TODO: recheck
def aggregate_for_norm(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    x_type: VariableType,
    y_type: VariableType,
) -> np.ndarray:
    if "continuous" in [x_type, y_type]:
        return df[loss_column].to_numpy()
    else:    
        grid = df.pivot_table(
            index=y_column, 
            columns=x_column,
            values=loss_column, 
            aggfunc="mean",
        )
        return grid.values.ravel()


def cluster_rows(
    grid: pd.DataFrame,
    method: Literal["mean", "correlation"],
) -> list[Any]:
    """
    Reorder rows of a pivot grid by mean or correlation clustering.
    Returns the list of index to order the grid rows with.
    """
    if len(grid) < 2:
        return grid.index.tolist()
    
    if method == "mean":
        return grid.mean(axis=1).sort_values().index.tolist()
    
    else:
        # pdist act on rows by default
        ## TODO: understand what correlation uses
        dist = pdist(grid.values, metric="correlation")
        Z = linkage(dist, method="average")
        return [grid.index[i] for i in leaves_list(Z)]
    


@dataclass
class SharedPlotState:
    norm: Normalize | TwoSlopeNorm | None = None
    row_order: list | None = None
    col_order: list | None = None


def compute_shared_state(
    df: pd.DataFrame,
    x_hp: str,
    y_hp: str,
    x_type: VariableType,
    y_type: VariableType,
    true_loss_column: str,
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"],
    cluster_by: Literal["mean", "correlation"],
    share_color_norm: bool,
    share_clustering: bool,
) -> SharedPlotState:
    state = SharedPlotState()

    if share_color_norm and normalize_color != "none":
        true_vals = aggregate_for_norm(df, x_hp, y_hp, true_loss_column, x_type, y_type)
        state.norm = build_norm(true_vals, normalize_color)

    if share_clustering and "nominal" in (x_type, y_type):
        true_grid = df.pivot_table(
            index=y_hp, 
            columns=x_hp,
            values=true_loss_column, 
            aggfunc="mean",
        )
        if y_type == "nominal":
            state.row_order = cluster_rows(true_grid, method=cluster_by)
        if x_type == "nominal":
            state.col_order = cluster_rows(true_grid.T, method=cluster_by)

    return state
