from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Literal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d

from metatab._paper.analysis.contour.utils import (
    apply_ordinal_order,
    build_norm,
    cluster_rows,
    SharedPlotState,
    compute_bin_edges
)



def plot_continuous_continuous(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    shared_state: None | SharedPlotState = None
) -> tuple[Axes, SharedPlotState]:
    '''
    Contour plot of loss over a continuous x continuous hyperparameter space.

    Since hyperparameter search produces scattered (x, y, loss) points with no
    guaranteed grid structure, the function first estimates loss on a regular
    100x100 grid via Delaunay triangulation (linear griddata), falling back to
    nearest neighbor outside the convex hull of the observations. The resulting
    10000-point matrix is then passed to contourf which traces filled regions
    between 50 equally spaced loss isolines, with black contour edges overlaid
    at 10 levels.

    Grid points outside the convex hull are filled by nearest neighbor and will
    appear as flat saturated regions at the plot edges.

    Returns the contourf mappable and the shared_state (input one if provided,
    fresh otherwise), carrying the norm for consistent color scaling across
    comparison plots.
    '''
    x = df[x_column].astype(float)
    y = df[y_column].astype(float)
    z = df[loss_column].astype(float)

    if log_x: x = np.log10(x)
    if log_y: y = np.log10(y)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    
    # build a share state object
    if shared_state is None:
        norm, ticks = build_norm(z.to_numpy(), normalize_color)
        shared_state = SharedPlotState(norm=norm, ticks=ticks)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    Zi = np.where(np.isnan(Zi), griddata((x, y), z, (Xi, Yi), method="nearest"), Zi)

    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap="coolwarm", norm=shared_state.norm)
    ax.contour(Xi, Yi, Zi, levels=10, colors="black", linewidths=0.5, alpha=0.4)

    if log_x:
        ticks = np.linspace(x.min(), x.max(), 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{10**t:.0e}" for t in ticks])
    if log_y:
        ticks = np.linspace(y.min(), y.max(), 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{10**t:.0e}" for t in ticks])

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    return cf, shared_state



def plot_nominal_nominal(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    cluster_by: Literal["mean", "correlation"] = "correlation",
    shared_state: SharedPlotState | None = None
) -> tuple[Axes, SharedPlotState]:
    grid = df.pivot_table(
        index=y_column, 
        columns=x_column,
        values=loss_column, 
        aggfunc="mean",
    )

    if shared_state is None:
        # cluster rows and columns indipendently
        col_order = cluster_rows(grid.T, method=cluster_by)
        row_order = cluster_rows(grid, method=cluster_by)
        norm, ticks = build_norm(grid.values.ravel(), normalize_color)
        shared_state = SharedPlotState(norm, row_order, col_order, ticks=ticks)
    
    grid: pd.DataFrame = grid.loc[shared_state.row_order, shared_state.col_order]

    # we use imshow since no meaningful notion of distance exists between nominal categories
    cf = ax.imshow(
        grid.values,
        aspect="auto", 
        origin="lower",
        cmap="coolwarm", 
        norm=shared_state.norm
    )

    ax.set_xticks(np.arange(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(np.arange(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return cf, shared_state



def plot_ordinal_ordinal(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    x_order: list | None = None,
    y_order: list | None = None,
    type_plot: Literal["heat", "contour"] = "heat",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    numeric: bool = True,
    shared_state: None | SharedPlotState = None
):
    """
    Respects an explicit ordinal order on both axes.
    x_order / y_order are lists of category values in the desired sequence.
    Missing categories are appended at the end.
    
    Allows for heatmaps and contours since the concept of 2d distance is valid.
    
    numeric: True use actual numeric values; False map to indices.
    """
    grid = df.pivot_table(
        index=y_column, 
        columns=x_column,
        values=loss_column, 
        aggfunc="mean",
    )

    # reorder axes according to ordinal order
    x_cats = apply_ordinal_order(grid.columns.tolist(), x_order)
    y_cats = apply_ordinal_order(grid.index.tolist(), y_order)
    grid = grid.loc[y_cats, x_cats]

    if shared_state is None:
        norm, ticks = build_norm(grid.values.ravel(), normalize_color)
        shared_state = SharedPlotState(norm=norm, ticks=ticks)

    if type_plot == "contour":
        Z = grid.values.astype(float)
        if numeric:
            # take directly the values
            xs = grid.columns.astype(float).to_numpy()
            ys = grid.index.astype(float).to_numpy()
            X, Y = np.meshgrid(xs, ys)
        else:
            # build the grid
            X, Y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))

        cf = ax.contourf(X, Y, Z, levels=50, cmap="coolwarm", norm=shared_state.norm)
        ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    
    else:
        # here we do NOT cluster since we there is natural order on both axes
        cf = ax.imshow(
            grid.values, 
            aspect="auto", 
            origin="lower",
            cmap="coolwarm", 
            norm=shared_state.norm
        )
    
    # ticks: in numeric mode the axis already has the right scale,
    # but we still want the original category labels on the ticks
    if numeric:
        ax.set_xticks(xs)
        ax.set_xticklabels(grid.columns.tolist())
        ax.set_yticks(ys)
        ax.set_yticklabels(grid.index.tolist())
    else:
        ax.set_xticks(np.arange(len(grid.columns)))
        ax.set_xticklabels(grid.columns.tolist())
        ax.set_yticks(np.arange(len(grid.index)))
        ax.set_yticklabels(grid.index.tolist())

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)    
    return cf, shared_state


def plot_nominal_ordinal(
    ax: Axes,
    df: pd.DataFrame,
    nominal_col: str,
    ordinal_col: str,
    loss_column: str,
    *,
    ordinal_order: list | None = None,
    cluster_by: Literal["mean", "correlation", "fix"] = "correlation",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    sigma: float = 0,
    shared_state: None | SharedPlotState = None,
):
    '''
    Visualize the interaction between a nominal and an ordinal hyperparameter
    as a heatmap.

    The nominal variable is placed on the rows and optionally reordered using
    clustering based on either the mean loss or the similarity of its loss profile 
    across ordinal values. The ordinal variable is placed on
    the columns and follows the user-specified ordinal order.

    Each cell represents the mean value of ``loss_column`` for a given
    (nominal, ordinal) combination. Optional Gaussian smoothing can be applied
    along the ordinal axis to emphasize trends.
    '''    
    # nominal on rows, ordinal on columns
    grid = df.pivot_table(
        index=nominal_col, 
        columns=ordinal_col,
        values=loss_column, 
        aggfunc="mean",
    )

    # cluster nominal rows or take the shared, sort ordinal columns
    if shared_state is None:
        norm, ticks = build_norm(grid.values.ravel(), normalize_color)
        nom_cats = grid.index.tolist() if cluster_by == "fix" else cluster_rows(grid, cluster_by)
        shared_state = SharedPlotState(norm=norm, row_order=nom_cats, ticks=ticks)

    ord_cats = apply_ordinal_order(grid.columns.tolist(), ordinal_order)
    grid: pd.DataFrame = grid.loc[shared_state.row_order, ord_cats]

    if sigma > 0:
        display = gaussian_filter1d(grid.values, axis=1, sigma=sigma)
    else:
        display = grid.values
    
    cf = ax.imshow(
        display, 
        aspect="auto", 
        origin="lower", 
        cmap="coolwarm", 
        norm=shared_state.norm
    )

    ax.set_xticks(np.arange(len(ord_cats)))
    ax.set_xticklabels(ord_cats)
    ax.set_yticks(np.arange(len(shared_state.row_order)))
    ax.set_yticklabels(shared_state.row_order)
    ax.set_xlabel(ordinal_col)
    ax.set_ylabel(nominal_col)
    return cf, shared_state


def plot_nominal_continuous(
    ax: Axes,
    df: pd.DataFrame,
    nominal_col: str,
    continuous_col: str,
    loss_column: str,
    *,
    log_cont: bool = False,
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    cluster_by: Literal["mean", "correlation", "fix"] = "correlation",
    sigma: float = 0,
    shared_state: SharedPlotState | None = None,
) -> tuple[Axes, SharedPlotState]:
    '''
    Renders the observed loss values as a 2D heatmap where:
    - y-axis: nominal categories (e.g. optimizer), ordered by similarity of their profiles (except fix that use fix positions)
    - x-axis: continuous hyperparameter (e.g. learning rate), with physically correct cell widths
    - color: loss value at each (category, hp_value) cell

    It reports the values you actually measured, with three purely visual adjustments:
    1. cell widths reflect true spacing of the continuous axis
    2. NaN holes are filled by linear interpolation from neighbors
    3. optional Gaussian smoothing reduces noise in the color transitions
    '''
    # pivot sort numbers in ascending order
    # we are relying on it since interp wants a sorted sequence
    grid = df.pivot_table(
        index=nominal_col, 
        columns=continuous_col, 
        values=loss_column, 
        aggfunc="mean"
    )

    if shared_state is None:
        norm, ticks = build_norm(grid.values.ravel(), mode=normalize_color)
        row_order = grid.index.tolist() if cluster_by == "fix" else cluster_rows(grid, cluster_by)
        shared_state = SharedPlotState(norm=norm, row_order=row_order, ticks=ticks)
    
    grid: pd.DataFrame = grid.loc[shared_state.row_order, :]

    # continuous axis
    x_cont = grid.columns.to_numpy(dtype=float)
    # we take the log before computing cell edges (widths)
    # in order to have them in log space in accordance to what is plotted
    if log_cont: x_cont = np.log10(x_cont)
    cont_edges = compute_bin_edges(x_cont)

    # we use interpolation ONLY to fill NaNs. 
    # We have 1500 base points so we have no need to densify.
    Z = np.vstack([
        np.interp(x_cont, x_cont[mask], row_vals[mask])
        for _, row_vals in grid.iterrows()
        for mask in [np.isfinite(row_vals.to_numpy(dtype=float))]
    ])

    # we computed the norm before smooting. 
    # It creates some discrepancy but acceptable in practice
    if sigma > 0:
        Z = gaussian_filter1d(Z, sigma=sigma, axis=1, mode="nearest")

    y_edges = np.arange(len(shared_state.row_order) + 1) - 0.5

    im = ax.pcolormesh(
        cont_edges,
        y_edges,
        Z,
        cmap="coolwarm",
        norm=shared_state.norm,
        shading="flat",
        rasterized=True,
    )

    tick_indices = np.linspace(0, len(x_cont) - 1, 5).astype(int)
    tick_pos = x_cont[tick_indices]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [f"{10**v:.2e}" if log_cont else f"{v:.4g}" for v in tick_pos]
    )
    ax.set_yticks(np.arange(len(shared_state.row_order)))
    ax.set_yticklabels(shared_state.row_order)
    
    ax.set_xlabel(continuous_col)
    ax.set_ylabel(nominal_col)
    
    return im, shared_state