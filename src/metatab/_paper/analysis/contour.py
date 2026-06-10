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

VariableType = Literal["nominal", "ordinal", "continuous"]


def _plot_continuous_continuous(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    type_plot: Literal["contour"] = "contour",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    shared_state: SharedPlotState
):
    if type_plot != "contour":
        raise ValueError("For continuous variables only 'contour' type_plot is possible.")

    x = df[x_column].astype(float)
    y = df[y_column].astype(float)
    z = df[loss_column].astype(float)

    if log_x: x = np.log10(x)
    if log_y: y = np.log10(y)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    norm = shared_state.norm \
        if shared_state.norm is not None \
        else build_norm(z.to_numpy(), normalize_color)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    Zi = np.where(np.isnan(Zi), griddata((x, y), z, (Xi, Yi), method="nearest"), Zi)

    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap="coolwarm", norm=norm)
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
    return cf



def _plot_nominal_nominal(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    type_plot: Literal["heat"] = "heat",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    cluster_by: Literal["mean", "correlation"] = "correlation",
    shared_state: SharedPlotState
):
    if type_plot != "heat":
        raise ValueError("'heat' type_plot is the only solution for nominal x nominal scenarios.")
    
    grid = df.pivot_table(
        index=y_column, 
        columns=x_column,
        values=loss_column, 
        aggfunc="mean",
    )

    # cluster rows and columns indipendently or take the shared order
    col_order = shared_state.col_order \
        if shared_state.col_order \
        else cluster_rows(grid.T, method=cluster_by)
    
    row_order = shared_state.row_order \
        if shared_state.row_order \
        else cluster_rows(grid, method=cluster_by)
    
    grid = grid.loc[row_order, col_order]

    norm = shared_state.norm \
        if shared_state.norm is not None \
        else build_norm(grid.values.ravel(), normalize_color)

    cf = ax.imshow(
        grid.values, 
        aspect="auto", 
        origin="lower",
        cmap="coolwarm", 
        norm=norm
    )

    ax.set_xticks(np.arange(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(np.arange(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    return cf



def _plot_ordinal_ordinal(
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
    shared_state: SharedPlotState
):
    """
    Like nominal_x_nominal but respects an explicit ordinal order on both axes.
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

    norm = shared_state.norm \
        if shared_state.norm  is not None \
        else build_norm(grid.values.ravel(), normalize_color)

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

        cf = ax.contourf(X, Y, Z, levels=50, cmap="coolwarm", norm=norm)
        ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    
    else:
        # here we do NOT cluster since we there is natural order on both axes
        cf = ax.imshow(
            grid.values, 
            aspect="auto", 
            origin="lower",
            cmap="coolwarm", 
            norm=norm
        )
    
    # ticks: in numeric mode the axis already has the right scale,
    # but we still want the original category labels on the ticks
    if numeric:
        ax.set_xticks(xs)
        ax.set_yticks(ys)
    else:
        ax.set_xticks(np.arange(len(grid.columns)))
        ax.set_yticks(np.arange(len(grid.index)))

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    return cf


def _plot_ordinal_nominal(
    ax: Axes,
    df: pd.DataFrame,
    loss_column: str,
    *,
    ordinal_col: str,
    nominal_col: str,
    ordinal_order: list | None = None,
    type_plot: Literal["heat"] = "heat",
    cluster_by: Literal["mean", "correlation"] = "correlation",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    sigma: float = 0,
    shared_state: SharedPlotState,
):
    if type_plot != "heat":
        raise ValueError("Only 'heat' type_plot is possible in ordinal x nominal")
    
    # nominal on rows, ordinal on columns
    grid = df.pivot_table(
        index=nominal_col, 
        columns=ordinal_col,
        values=loss_column, 
        aggfunc="mean",
    )

    # cluster nominal rows or take the shared, sort ordinal columns
    nom_cats = shared_state.row_order if shared_state.row_order else cluster_rows(grid, method=cluster_by)
    ord_cats = apply_ordinal_order(grid.columns.tolist(), ordinal_order)
    grid = grid.loc[nom_cats, ord_cats]
    
    if sigma > 0:
        display = gaussian_filter1d(grid.values, axis=1, sigma=sigma)
    else:
        display = grid.values

    norm = shared_state.norm if shared_state.norm else build_norm(display.ravel(), normalize_color)
    cf = ax.imshow(display, aspect="auto", origin="lower", cmap="coolwarm", norm=norm)

    ax.set_xticks(np.arange(len(ord_cats)))
    ax.set_xticklabels(ord_cats)
    ax.set_yticks(np.arange(len(nom_cats)))
    ax.set_yticklabels(nom_cats)
    ax.set_xlabel(ordinal_col)
    ax.set_ylabel(nominal_col)
    return cf


# def _plot_nominal_continuous(
#     ax: Axes,
#     df: pd.DataFrame,
#     loss_column: str,
#     *,
#     nominal_col: str,
#     continuous_col: str,
#     log_cont: bool = False,
#     type_plot: Literal["contour"] = "contour",
#     normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
#     cluster_by: Literal["mean", "correlation"] = "correlation",
#     sigma: float = 0,
#     shared_state: SharedPlotState,
# ):
#     """
#     One axis is nominal (categories), the other is continuous.
#     For each category, the loss is interpolated over the continuous range
#     onto a shared 1-D grid, producing a 2-D heatmap (categories × cont grid).
#     Nominal rows are then clustered by correlation of their interpolated profiles.
#     """

#     if type_plot == "heat":
#         raise ValueError("heat type_plot not possible for nominal x continuos combination")
    
#     # --- 1. Build shared continuous grid ---
#     cont_vals = df[continuous_col].astype(float)
#     if log_cont:
#         cont_vals = np.log10(cont_vals)

#     xi = np.linspace(cont_vals.min(), cont_vals.max(), 100)
#     categories = df[nominal_col].unique().tolist()

#     # --- 2. Interpolate each category onto the shared grid ---
#     heatmap = np.full((len(categories), len(xi)), np.nan)

#     for row_idx, cat in enumerate(categories):
#         subset = df.loc[df[nominal_col] == cat]
#         c = subset[continuous_col].astype(float)
#         if log_cont: c = np.log10(c)
#         loss = subset[loss_column].astype(float)

#         mask = np.isfinite(c) & np.isfinite(loss)
#         c, loss = c[mask].to_numpy(), loss[mask].to_numpy()

#         order = np.argsort(c)
#         c_s, l_s = c[order], loss[order]

#         interp = griddata(c_s, l_s, xi, method="linear")
#         nearest = griddata(c_s, l_s, xi, method="nearest")
#         heatmap[row_idx] = np.where(np.isnan(interp), nearest, interp)


#     # Cluster nominal rows by correlation of their profiles ---
#     heatmap_df = pd.DataFrame(heatmap, index=categories)

#     row_order = (
#         shared_state.row_order
#         if shared_state.row_order
#         else cluster_rows(heatmap_df, method=cluster_by)
#     )

#     # reorder heatmap rows to match clustering
#     row_idx_map = {cat: i for i, cat in enumerate(categories)}
#     ordered_indices = [row_idx_map[cat] for cat in row_order]
#     heatmap = heatmap[ordered_indices]

#     # Norm ---
#     norm = (
#         shared_state.norm
#         if shared_state.norm is not None
#         else build_norm(heatmap.ravel(), normalize_color)
#     )

#     # Render ---
#     X, Y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
#     cf = ax.contourf(X, Y, heatmap, levels=50, cmap="coolwarm", norm=norm)
#     ax.contour(X, Y, heatmap, levels=10, colors="black", linewidths=0.5, alpha=0.4)

#     # x-ticks: continuous axis labels
#     n_ticks = 5
#     tick_indices = np.linspace(0, len(xi) - 1, n_ticks).astype(int)
#     tick_vals = xi[tick_indices]
#     ax.set_xticks(tick_indices)
#     ax.set_xticklabels([f"{10**v:.2e}" if log_cont else f"{v:.4g}" for v in tick_vals])

#     # y-ticks: nominal category labels in clustered order
#     ax.set_yticks(np.arange(len(row_order)))
#     ax.set_yticklabels(row_order)

#     ax.set_xlabel(continuous_col)
#     ax.set_ylabel(nominal_col)
#     return cf



# ============================================================
# Dispatcher
# ============================================================

def _plot_hp_surface(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    x_type: VariableType = "continuous",
    y_type: VariableType = "continuous",
    x_order: list | None = None,   # ordinal order for x axis
    y_order: list | None = None,   # ordinal order for y axis
    cluster_by: Literal["mean", "correlation"] = "correlation",
    log_x: bool = False,
    log_y: bool = False,
    type_plot: Literal["heat", "contour"] = "heat",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    sigma: float = 0,
    shared_state: SharedPlotState
):
    # ordinal × ordinal
    if x_type == "ordinal" and y_type == "ordinal":
        return _plot_ordinal_ordinal(
            ax, 
            df, 
            x_column, 
            y_column, 
            loss_column,
            x_order=x_order, 
            y_order=y_order,
            type_plot=type_plot,
            normalize_color=normalize_color,
            shared_state=shared_state
        )

    # ordinal × nominal (the opposite direction is forbidden by earlier check)
    elif x_type == "ordinal" and y_type == "nominal":
        return _plot_ordinal_nominal(
            ax, 
            df,  
            loss_column,
            ordinal_col=x_column,
            nominal_col=y_column,
            ordinal_order=y_order,
            cluster_by=cluster_by,
            type_plot=type_plot,
            normalize_color=normalize_color,
            sigma=sigma,
            shared_state=shared_state
        )

    # nominal × nominal
    elif x_type == "nominal" and y_type == "nominal":
        return _plot_nominal_nominal(
            ax, 
            df, 
            x_column, 
            y_column, 
            loss_column,
            type_plot=type_plot,
            normalize_color=normalize_color,
            cluster_by=cluster_by,
            shared_state=shared_state
        )

    # continuous × continuous
    elif x_type == "continuous" and y_type == "continuous":
        return _plot_continuous_continuous(
            ax, 
            df, 
            x_column, 
            y_column, 
            loss_column,
            log_x=log_x, 
            log_y=log_y,
            type_plot=type_plot,
            normalize_color=normalize_color,
            shared_state =shared_state
        )

    # the opposite direction is forbidden by earlier check
    elif x_type == "continuos" and y_type == "nominal":
        return _plot_nominal_continuous(
                ax,
                df,
                loss_column,
                nominal_col=y_column,
                continuous_col=x_column,
                log_cont=log_y,
                type_plot=type_plot,
                normalize_color=normalize_color,
                cluster_by=cluster_by,
                sigma=sigma,
                shared_state=shared_state,
            )
    else:
        raise ValueError("Combination of xy types not admissed.")



# ============================================================
# Public API
# ============================================================

def plot_dataset_comparison(
    ax_true: Axes,
    ax_pred: Axes,
    ax_uncertainty: Axes | None,
    df: pd.DataFrame,
    x_hp: str,
    y_hp: str,
    log_x: bool = False,
    log_y: bool = False,
    y_type: VariableType = "continuos",
    x_type: VariableType = "continuos",
    x_order: list | None = None,
    y_order: list | None = None,
    true_loss_column: str = "z_normalized_loss",
    pred_loss_column: str = "pred_z_normalized_loss",
    uncertainty_column: str = "uncertainty",
    title_suffix: str = "",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    type_plot: Literal["heat", "contour"] = "heat",
    cluster_by: Literal["mean", "correlation"] = "correlation",
    share_color_norm: bool = True,
    cbar_ax: Axes | None = None,
    sigma: float = 0,
    share_clustering: bool = True,
):
    if x_type not in ['nominal', 'ordinal', 'continuous']:
        raise ValueError("Wrong 'x_type'.")
    
    if y_type not in ['nominal', 'ordinal', 'continuous']:
        raise ValueError("Wrong 'y_type'.")
    
    if x_type == "nominal" and y_type != "nominal":
        raise ValueError("x_type nominal is possible only with a y_type nominal. Set it on the y.")

    shared_state = compute_shared_state(
        df=df,
        x_hp=x_hp,
        y_hp=y_hp,
        x_type=x_type,
        y_type=y_type,
        true_loss_column=true_loss_column,
        normalize_color=normalize_color,
        cluster_by=cluster_by,
        share_color_norm=share_color_norm,
        share_clustering=share_clustering
    )

    def plot(ax, loss_column, title):
        cf = _plot_hp_surface(
            ax, df, x_hp, y_hp, loss_column,
            x_type=x_type, 
            y_type=y_type,
            x_order=x_order, 
            y_order=y_order,
            cluster_by=cluster_by,
            log_x=log_x, 
            log_y=log_y,
            type_plot=type_plot,
            normalize_color=normalize_color,
            sigma=sigma,
            shared_state=shared_state,
        )
        ax.set_title(f"{title} - {title_suffix}")
        return cf

    cf_true = plot(ax_true, true_loss_column, "Actual")
    cf_pred = plot(ax_pred, pred_loss_column, "Predicted")

    if shared_state.norm is not None:
        plt.colorbar(cf_true, cax=cbar_ax, label="Loss") \
            if cbar_ax is not None \
            else plt.colorbar(cf_true, ax=[ax_true, ax_pred], label="Loss")
    else:
        plt.colorbar(cf_true, ax=ax_true, label="Actual Loss")
        plt.colorbar(cf_pred, ax=ax_pred, label="Predicted Loss")

    if ax_uncertainty is not None:
        cf = _plot_hp_surface(
            ax_uncertainty, 
            df, 
            x_hp, 
            y_hp, 
            uncertainty_column,
            x_type=x_type, 
            y_type=y_type,
            x_order=x_order, 
            y_order=y_order,
            cluster_by=cluster_by,
            log_x=log_x, 
            log_y=log_y,
            type_plot=type_plot,
        )
        plt.colorbar(cf, ax=ax_uncertainty, label="Uncertainty")
        ax_uncertainty.set_title(f"Uncertainty - {title_suffix}")