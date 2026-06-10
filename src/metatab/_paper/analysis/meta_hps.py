from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.axes import Axes
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ============================================================
# Utilities
# ============================================================

def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _build_norm(
    a: np.ndarray,
    mode: Literal["none", "min_max", "quantile", "centered_0"], 
    qrange=(2, 98)
):
    if mode == "none":
        return None

    elif mode == "min_max":
        return Normalize(vmin=a.min(), vmax=a.max())

    elif mode == "quantile":
        vmin, vmax = np.percentile(a, qrange)
        return Normalize(vmin=vmin, vmax=vmax)

    # we take the percentiles as limits
    elif mode == "centered_0":
        vmax = np.percentile(a, qrange[1])
        vmin = np.percentile(a, qrange[0])
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    
    else:
        raise ValueError(f"Unknown normalize_color: {mode}")


def _aggregate_for_norm(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    is_x_cat: bool,
    is_y_cat: bool,
) -> np.ndarray:
    """
    Returns the values that will actually be rendered in the colormap,
    so that _build_norm operates on the same data the plot uses.
 
    - cat x cat  → pivot_table means (mirrors _plot_categorical_categorical)
    - everything else → raw column values (no aggregation happens before plotting)
    """
    if is_x_cat and is_y_cat:
        grid = df.pivot_table(
            index=y_column,
            columns=x_column,
            values=loss_column,
            aggfunc="mean",
        )
        return grid.values.ravel()
    else:
        return df[loss_column].to_numpy()
    


# ============================================================
# Continuous x Continuous
# ============================================================

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
    norm_override: None | Normalize | TwoSlopeNorm = None
):
    if type_plot != "contour":
        raise ValueError("For continuos variables only 'contour' type_plot is possible.")
    
    x = df[x_column].astype(float)
    y = df[y_column].astype(float)
    z = df[loss_column].astype(float)

    if log_x: x = np.log10(x)
    if log_y: y = np.log10(y)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    
    norm = norm_override \
        if norm_override is not None \
        else _build_norm(z.to_numpy(), normalize_color)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    Zi_nearest = griddata((x, y), z, (Xi, Yi), method="nearest")
    Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    #Zi = gaussian_filter(Zi, sigma=1.0)

    cf = ax.contourf(
        Xi,
        Yi,
        Zi,
        levels=50,
        cmap="coolwarm",
        norm=norm
    )

    ax.contour(
        Xi,
        Yi,
        Zi,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.4,
    )

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


# ============================================================
# Categorical x Categorical
# ============================================================

def _plot_categorical_categorical(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    type_plot: Literal["heat", "contour"] = "heat",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    norm_override: None | Normalize | TwoSlopeNorm = None
):
    '''
    The function treat the categoricals variables as nominal,
    meaning it convert them to index and the index is the position on the plot.
    This fails in case in which the categories are numeric with different distances
    between them (i.e 1, 2, 10 -> get mapped to same index with same distances).
    '''
    grid = df.pivot_table(
        index=y_column,
        columns=x_column,
        values=loss_column,
        aggfunc="mean",
    )

    norm = norm_override \
        if norm_override is not None \
        else _build_norm(grid.values.ravel(), normalize_color)

    if type_plot == "contour":
        Z = grid.values
        X, Y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
        
        cf = ax.contourf(
            X,
            Y,
            Z,
            levels=50,
            cmap="coolwarm",
            norm=norm
        )

        ax.contour(
            X,
            Y,
            Z,
            levels=10,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )

    else:
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


# ============================================================
# Mixed
# ============================================================
def _plot_mixed(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    loss_column: str,
    *,
    is_x_cat: bool,
    is_y_cat: bool,
    log_x: bool = False,
    log_y: bool = False,
    type_plot: Literal["heat", "contour"] = "heat",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    norm_override: None | Normalize | TwoSlopeNorm = None,
):
    # set orientation
    if is_x_cat:
        cat_col, cont_col, log_cont = x_column, y_column, log_y
        categorical_on_x = True
    else:
        cat_col, cont_col, log_cont = y_column, x_column, log_x
        categorical_on_x = False

    categories = sorted(df[cat_col].unique())
    cont_all = df[cont_col].astype(float)
    if log_cont: cont_all = np.log10(cont_all)

    xi = np.linspace(cont_all.min(), cont_all.max(), 100)
    heatmap = np.full((len(categories), len(xi)), np.nan)

    # fill heatmap row by row (rows=categories, cols=continuous)
    for row_idx, cat in enumerate(categories):
        subset = df.loc[df[cat_col] == cat, :]
        c = subset[cont_col].astype(float)
        if log_cont: c = np.log10(c)
        loss = subset[loss_column].astype(float)

        # sort required for interpolation
        order = np.argsort(c.to_numpy())
        c_sorted = c.to_numpy()[order]
        loss_sorted = loss.to_numpy()[order]

        # linear interpolation, nearest for extrapolated edges (mirrors griddata behavior)
        interpolated = griddata(c_sorted, loss_sorted, xi, method="linear")
        nearest = griddata(c_sorted, loss_sorted, xi, method="nearest")
        heatmap[row_idx] = np.where(np.isnan(interpolated), nearest, interpolated)

    norm = norm_override if norm_override is not None else _build_norm(heatmap.ravel(), normalize_color)

    # transpose when categorical is on x so that:
    # categorical_on_x=True  → plot_data shape (len(xi), n_categories): x=categories, y=continuous
    # categorical_on_x=False → plot_data shape (n_categories, len(xi)): x=continuous, y=categories
    heatmap = heatmap.T if categorical_on_x else heatmap

    if type_plot == "contour":
        X, Y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
        cf = ax.contourf(X, Y, heatmap, levels=50, cmap="coolwarm", norm=norm)
        ax.contour(X, Y, heatmap, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    else:
        cf = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="coolwarm", norm=norm)

    cont_vals = np.linspace(cont_all.min(), cont_all.max(), 5)
    cont_labels = [f"{10**v:.2e}" if log_cont else f"{v:.2f}" for v in cont_vals]

    if categorical_on_x:
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories)
        cont_ticks = np.linspace(0, heatmap.shape[0] - 1, 5)
        ax.set_yticks(cont_ticks)
        ax.set_yticklabels(cont_labels)
    else:
        ax.set_yticks(np.arange(len(categories)))
        ax.set_yticklabels(categories)
        cont_ticks = np.linspace(0, heatmap.shape[1] - 1, 5)
        ax.set_xticks(cont_ticks)
        ax.set_xticklabels(cont_labels)

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return cf



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
    is_x_cat: bool = False,
    is_y_cat: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    type_plot: Literal["heat", "contour"] = "heat",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    norm_override: None | Normalize | TwoSlopeNorm = None
):
    if is_x_cat and is_y_cat:
        return _plot_categorical_categorical(
            ax,
            df,
            x_column,
            y_column,
            loss_column,
            type_plot=type_plot,
            normalize_color=normalize_color,
            norm_override=norm_override
        )

    elif not is_x_cat and not is_y_cat:
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
            norm_override=norm_override
        )

    else:
        return _plot_mixed(
            ax,
            df,
            x_column,
            y_column,
            loss_column,
            is_x_cat=is_x_cat,
            is_y_cat=is_y_cat,
            log_x=log_x,
            log_y=log_y,
            type_plot=type_plot,
            norm_override=norm_override
        )
    


def plot_dataset_comparison(
    ax_true: Axes,
    ax_pred: Axes,
    ax_uncertainty: Axes | None,
    df: pd.DataFrame,
    x_hp: str,
    y_hp: str,
    true_loss_column: str = "z_normalized_loss",
    pred_loss_column: str = "pred_z_normalized_loss",
    uncertainty_column: str = "uncertainty",
    title_suffix: str = "",
    normalize_color: Literal["none", "min_max", "centered_0", "quantile"] = "min_max",
    type_plot: Literal["heat", "contour"] = "heat",
    log_x: bool = False,
    log_y: bool = False,
    is_x_cat: bool | Literal["infer"] = "infer",
    is_y_cat: bool | Literal["infer"] = "infer",
    share_color_norm: bool = True,
    cbar_ax: Axes | None = None
):
    if isinstance(is_x_cat, str): 
        is_x_cat = _is_categorical(df[x_hp])

    if isinstance(is_y_cat, str): 
        is_y_cat = _is_categorical(df[y_hp])

    # build the shared norm when requested
    shared_norm = None
    if share_color_norm and normalize_color != "none":
        true_vals = _aggregate_for_norm(
            df, x_hp, y_hp, true_loss_column, is_x_cat, is_y_cat
        )
        shared_norm = _build_norm(true_vals, normalize_color)

    def plot(ax, loss_column, title):
        cf = _plot_hp_surface(
            ax,
            df,
            x_hp,
            y_hp,
            loss_column,
            log_x=log_x,
            log_y=log_y,
            type_plot=type_plot,
            normalize_color=normalize_color,
            is_x_cat=is_x_cat,
            is_y_cat=is_y_cat,
            norm_override=shared_norm
        )
        ax.set_title(f"{title} - {title_suffix}")
        return cf

    cf_true = plot(ax_true, true_loss_column, "Actual")
    cf_pred = plot(ax_pred, pred_loss_column, "Predicted")

    if shared_norm is not None:
        # single colorbar spanning both axes or use the provded
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
            log_x=log_x,
            log_y=log_y,
            type_plot=type_plot,
            is_x_cat=is_x_cat,
            is_y_cat=is_y_cat,
        )
        plt.colorbar(cf, ax=ax_uncertainty, label="Uncertainty")
        ax_uncertainty.set_title(f"Uncertainty - {title_suffix}")




### continuos x continuos (DEPRECATED)
# def plot_best_percentile_points(
#     ax,
#     df,
#     x_column,
#     y_column,
#     loss_column="loss",
#     percentile=25,
#     log_x=False,
#     log_y=False,
# ):
#     x = np.log10(df[x_column]) if log_x else df[x_column]
#     y = np.log10(df[y_column]) if log_y else df[y_column]
#     z = df[loss_column].to_numpy()

#     mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
#     x, y, z = x[mask], y[mask], z[mask]

#     threshold = np.percentile(z, percentile)
#     best_mask = z <= threshold

#     ax.scatter(
#         x[~best_mask],
#         y[~best_mask],
#         color="lightgray",
#         s=15,
#         alpha=0.3,
#         label=f"Bottom {100 - percentile:.0f}%"
#     )

#     ax.scatter(
#         x[best_mask],
#         y[best_mask],
#         color="red",
#         s=25,
#         alpha=0.8,
#         label=f"Top {percentile:.0f}%"
#     )

#     ax.set_xlabel(x_column)
#     ax.set_ylabel(y_column)
#     ax.legend()

#     return ax