from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Literal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d

from metatab._paper.analysis.utils_contour import (
    apply_ordinal_order,
    build_norm,
    cluster_rows,
    compute_shared_state,
    SharedPlotState
)


def plot_nominal_continuous_surface(
    df: pd.DataFrame,
    nominal_col: str,
    continuous_col: str,
    loss_column: str,
    *,
    log_cont: bool = False,
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    cluster_by: Literal["mean", "correlation"] = "correlation",
    sigma: float = 0,
    figsize: tuple[float, float] | None = None,
    title: str = "",
    shared_state: SharedPlotState | None = None,
) -> tuple[plt.Figure, np.ndarray]:

    # --- 1. Pivot & cluster row order ---
    lasagna_df = df.pivot_table(
        index=nominal_col, 
        columns=continuous_col, 
        values=loss_column, 
        aggfunc="mean"
    )

    ## TODO die here
    # row_order = (
    #     shared_state.row_order
    #     if shared_state and shared_state.row_order
    #     else cluster_rows(lasagna_df, method=cluster_by)
    # )
    
    row_order = lasagna_df.index.tolist()
    lasagna_df = lasagna_df.loc[row_order]

    # --- 2. Continuous axis (log optional) ---
    epvs = lasagna_df.columns.to_numpy(dtype=float)
    if log_cont: epvs = np.log10(epvs)
    epv_edges = _compute_bin_edges(epvs)


    # --- 3. Interpolate each row ---
    Z = np.vstack([
        np.interp(epvs, epvs[mask], row_vals[mask])
        for _, row_vals in lasagna_df.iterrows()
        for mask in [np.isfinite(row_vals.to_numpy(dtype=float))]
    ])

    if sigma > 0:
        Z = gaussian_filter1d(Z, sigma=sigma, axis=1, mode="nearest")

    # --- 4. Norm ---
    norm = (
        shared_state.norm
        if shared_state and shared_state.norm is not None
        else build_norm(Z.ravel(), normalize_color)
    )

    # --- 5. Layout ---
    y_edges = np.arange(len(row_order) + 1) - 0.5
    figsize = figsize or (12, max(4, len(row_order) * 0.5))
    fig, ax_heat = plt.subplots(figsize=figsize)

    # --- 6. pcolormesh ---
    im = ax_heat.pcolormesh(
        epv_edges,
        y_edges,
        Z,
        cmap="coolwarm",
        norm=norm,
        shading="flat",
        rasterized=True,
    )

    tick_indices = np.linspace(0, len(epvs) - 1, 5).astype(int)
    tick_pos = epvs[tick_indices]
    ax_heat.set_xticks(tick_pos)
    ax_heat.set_xticklabels(
        [f"{10**v:.2e}" if log_cont else f"{v:.4g}" for v in tick_pos]
    )
    ax_heat.set_yticks(np.arange(len(row_order)))
    ax_heat.set_yticklabels(row_order)
    ax_heat.set_xlabel(continuous_col)
    ax_heat.set_ylabel(nominal_col)
    ax_heat.set_title(title)
    plt.colorbar(im, ax=ax_heat, label=loss_column, fraction=0.03, pad=0.02)

    return fig, Z



def _compute_bin_edges(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges
