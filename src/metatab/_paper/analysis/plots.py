import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal
from matplotlib.axes import Axes
from adjustText import adjust_text

from metatab._paper.analysis.utils import (
    check_presence_cols, 
    append_if_not_none,
    ensure_or_create
)



def draw_scatterplot_performance_runtime(
    ax: Axes, 
    df: pd.DataFrame,
    performance_column: str,
    runtime_column: str,
    performance_std_column: str | None = None,
    runtime_std_column: str | None = None,
    hue_column: str | None = None,
    style_column: str | None = None,
    palette: None | dict = None,
    map_style: None | dict = None,
    label_column: str | None = None,
    error_bar_args: dict | None = None,
    sns_scatterplot_args: dict | None = None,
) -> Axes:
    '''
    Plot the performance-runtime scatterplot on the input Axes.
    Labels the points with the text in "label_column" column.

    Parameters:
        ax (Axes): Axes onto which draw the plot.
        
        df (pd.DataFrame): Dataframe containing the data to plot.
        
        performance_column (str): Name of the column containing the performance info.
        
        runtime_column (str): Name of the column containing the runtime info.
        
        performance_std_column(str | None, optional): 
            Name of the column containing the performance std info.
            If None no performance error bar is plotted.
        
        runtime_std_column (str | None, optional):
            Name of the column containing the runtime std info.
            If None no runtime error bar is plotted.

        hue_column (str | None, optional): 
            Name of the column to map as color. Can be None.

        style_column (str | None, optional):
            Name of the column to map to shapes. Can be None.

        map_style (dict | None, optional): 
            Map of shapes to use for the shape column.
        
        palette (dict | None, optional): 
            Dict of color mappings. If None the error bars (if any) will be gray.
        
        label_column (str | None, optional): 
            Name of the column with point labels info. Can be None.

        error_bar_args (dict | None, optional):
            Dict unpackaged in the "errorbar" matplotlib function.
        
        sns_scatterplot_args (dict | None, optional):
            Dict unpackaged in the "scatterplot" seaborne function.

    Returns:
        Axes: The axes.
    '''
    cols = [performance_column, runtime_column]

    for col in [label_column, hue_column, style_column]:
        cols = append_if_not_none(cols, col)

    check_presence_cols(df, cols)

    sns_scatterplot_args = ensure_or_create(sns_scatterplot_args, dict)
    error_bar_args = ensure_or_create(error_bar_args, dict)

    ax = sns.scatterplot(
        data=df, 
        x=performance_column,
        y=runtime_column,
        hue=hue_column,
        style=style_column,
        palette=palette,
        markers=map_style if map_style else True,
        ax=ax,
        **sns_scatterplot_args
    )

    if label_column:
        texts = []
        for _, row in df.iterrows():
            texts.append(
                ax.text(
                    row[performance_column],
                    row[runtime_column],
                    row[label_column]
                )
        )
        _ = adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))


    if hue_column is not None:
        for hue_category, df_hue_category in df.groupby(hue_column):
            color = "gray" \
                if palette is None or hue_category not in palette.keys() \
                else palette[hue_category]

            if performance_std_column:
                _ = ax.errorbar(
                    x=df_hue_category[performance_column],
                    y=df_hue_category[runtime_column],
                    xerr=df_hue_category[performance_std_column],
                    fmt="none",
                    ecolor=color,
                    **error_bar_args
                )
            
            if runtime_std_column:
                _ = ax.errorbar(
                    x=df_hue_category[performance_column],
                    y=df_hue_category[runtime_column],
                    yerr=df_hue_category[runtime_std_column],
                    fmt="none",
                    ecolor=color,
                    **error_bar_args
                )

    return ax



def draw_scatter_diagonal_plot(
    ax: Axes,
    df: pd.DataFrame,
    performance_column_y: str,
    performance_column_x: str,
    hue_column: str | None = None,
    style_column: str | None = None,
    palette: dict | None = None,
    map_style: dict | None = None,
    top_left_anno: str | None = None,
    bottom_right_anno: str | None = None,
    sns_scatterplot_args: dict | None = None
) -> Axes:
    '''
    Plot paired xy values in a scatter plot with a diagonal 
    line representing equal values.

    Parameters:
        ax (Axes): Axes onto which draw the plot.

        df (pd.DataFrame): Dataframe containing the data to plot.

        performance_column_y (str): Name of performance column put on the y axis.

        performance_column_x (str): Name of performance column put on the x axis.

        hue_column (str | None, optional): Name of the column mapped to hue.

        style_column (str | None, optional): Name of the column mapped to shape.

        palette (dict | None, optional): Palette of colors to use for the hue column.

        map_style (dict | None, optional): Map of shapes to use for the shape column.

        top_left_anno (str | None, optional): 
            Text annotation to report on the top left corner of the plot

        bottom_right_anno (str | None, optional):
            Text annotation to report on the down right corner of the plot.

        sns_scatterplot_args (dict, optional):
            Dict unpackaged in the "scatterplot" seaborne function.
    
    Returns:
        Axes: The axes.
    '''
    cols = [performance_column_x, performance_column_y]

    for col in [hue_column, style_column]:
        cols = append_if_not_none(cols, col)

    check_presence_cols(df, cols)

    sns_scatterplot_args = ensure_or_create(sns_scatterplot_args, dict)

    ax = sns.scatterplot(
        data=df, 
        x=performance_column_x,
        y=performance_column_y,
        hue=hue_column,
        style=style_column,
        palette=palette,
        markers=map_style if map_style else True,
        ax=ax,
        **sns_scatterplot_args
    )
    
    ax.plot([0, 1], [0, 1], color='black', linestyle='-', transform=ax.transAxes)

    if top_left_anno is not None:
        ax.text(
            x=0.02, 
            y=0.98, 
            s=top_left_anno, 
            transform=ax.transAxes,
            ha='left', 
            va='top',
            fontsize=10
        )

    if bottom_right_anno is not None:
        ax.text(
            x=0.98, 
            y=0.02, 
            s=bottom_right_anno, 
            transform=ax.transAxes,
            ha='right', 
            va='bottom',
            fontsize=10
        )

    return ax



def draw_win_heatmap_plot(
    ax: Axes,
    df: pd.DataFrame,
    delta_for_win: float = 0.0,
    ignore_draws: bool = False,
    na_strategy: Literal["error", "draw"] = "error",
    decimal_digits: int = 1,
    sns_heatmap_args: dict | None = None
) -> Axes:
    '''
    Draw a heatmap showing pairwise win percentages between methods.

    Each column in `df` represents a method (e.g., an estimator), 
    and each row represents a comparison case.
    For each pair of methods (A, B), the heatmap shows the percentage 
    of cases where A outperforms B by at least `delta_for_win`.

    The diagonal is set to nan since a method can't compare to itself.

    Parameters:
        ax (Axes): Axes onto which draw the heatmap.
        
        df (pd.DataFrame): 
            DataFrame where each column is a method and each row contains comparable values
        
        delta_for_win (float, optional): 
            Minimum difference (A - B) required to count as a win for A.
            Differences smaller than this threshold are treated as draws.
            If different from zero then the two complementar values may 
            not sum to 100 depending on the value of `ignore_draws`.

        ignore_draws (bool, optional):
            Whether to remove draws from percentages computations.
            If True assures that the heatmap remains complementary
            (this depends also on `na_strategy` in presence of nan).
        
        na_strategy (Literal["error", "count", "ignore"], optional):
            Strategy to apply in presence of na:
            - "error": raise an error
            - "count": consider comparisons involving nan in the denominator of percentage computation.
            - "ignore": remove comparisons involving nan from the percentage computation.

        decimal_digits (int, 1):
            Number of decimal digits to round the win percentages.

        sns_heatmap_args (dict | None, optional):
            Dict unpackaged in the "heatmap" seaborne function.

    Returns:
        Axes: The axes.
    '''
    if df.isna().any().any() and na_strategy == "error":
        raise ValueError("DataFrame contains na values.")
    
    sns_heatmap_args = ensure_or_create(sns_heatmap_args, dict)
    methods = df.columns
    n_methods = methods.size
    win_matrix = np.zeros((n_methods, n_methods))
    
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = df.iloc[:, i] - df.iloc[:, j]
            na_mask = pd.isna(diff)
            valid_diff = diff[~na_mask]
            
            # these 3 lines work and return all 0 when valid_diff size is 0 (all nan columns)
            wins_i = (valid_diff >= delta_for_win).sum()
            draws = (valid_diff.abs() < delta_for_win).sum()
            wins_j = valid_diff.size - (wins_i + draws)

            if na_strategy == "ignore":
                # only count valid comparisons
                den = valid_diff.size
            else:
                # count in denominator
                den = df.shape[0]

            if ignore_draws:
                den -= draws
            
            # if den goes to 0 then draws + nan = all
            pct_win_i = wins_i / den * 100 if den > 0 else 0
            pct_win_j = wins_j / den * 100 if den > 0 else 0
            
            win_matrix[i, j] = pct_win_i
            win_matrix[j, i] = pct_win_j


    # order by increasing average win-rate 
    average_wins = win_matrix.mean(axis=1)
    idx_sort = np.argsort(average_wins, stable=True)
    win_matrix = win_matrix[idx_sort, :][:, idx_sort]
    sorted_methods = methods[idx_sort]
    
    np.fill_diagonal(win_matrix, np.nan)
    df_heat = pd.DataFrame(win_matrix, columns=sorted_methods, index=sorted_methods)

    sns.heatmap(
        data=df_heat,
        annot=True,
        vmin=0, 
        vmax=100,
        fmt=f".{decimal_digits}f",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
        **sns_heatmap_args
    )
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax