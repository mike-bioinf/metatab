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
) -> tuple[None | Normalize | TwoSlopeNorm, np.ndarray | None]:
    '''
    Return a tuple with:
    1. The color normalizer or None when not requested.
    2. The ticks position as a 1D array, when mode == "quantile", or None.
    '''
    a = a[np.isfinite(a)]

    if len(a) == 0 or mode == "none":
        return None, None
    
    elif mode == "min_max":
        return Normalize(vmin=a.min(), vmax=a.max()), None
    
    elif mode == "quantile":
        vmin, vmax = np.percentile(a, qrange)
        return Normalize(vmin=vmin, vmax=vmax), None
    
    elif mode == "centered_0":
        vmax = np.percentile(a, qrange[1])
        vmin = np.percentile(a, qrange[0])
        #vmin, vmax = a.min(), a.max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        ticks = norm.inverse(np.linspace(0, 1, 9))
        return norm, ticks
    
    raise ValueError(f"Unknown normalize_color: {mode}")



def cluster_rows(
    grid: pd.DataFrame,
    method: Literal["mean", "correlation"],
) -> list[Any]:
    """
    Reorder rows of a pivot grid by mean or correlation clustering.
    Returns the list of index to order the grid rows with.
    """
    # return the the actual order for 2 or less categories
    if grid.shape[0] <= 2:
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
    ticks: np.ndarray | None = None
    row_order: list | None = None
    col_order: list | None = None


def compute_bin_edges(x):
    '''
    Compute the cell edges for each value in the array
    by getting the half distance from its neighbours.
    '''
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges
