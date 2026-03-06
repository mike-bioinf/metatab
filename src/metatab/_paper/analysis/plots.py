import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal
from matplotlib.axes import Axes
from adjustText import adjust_text
from paretoset import paretoset

from metatab._paper.analysis.utils import (
    check_presence_cols, 
    append_if_not_none,
    ensure_or_create
)



def draw_scatterplot(
    ax: Axes, 
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_std_column: str | None = None,
    y_std_column: str | None = None,
    hue_column: str | None = None,
    style_column: str | None = None,
    palette: None | dict = None,
    map_style: None | dict = None,
    label_column: str | None = None,
    show_pareto_frontier: bool = False,
    error_bar_args: dict | None = None,
    sns_scatterplot_args: dict | None = None,
    sns_lineplot_frontier_args: dict | None = None
) -> Axes:
    '''
    Plot the scatterplot on the input Axes.
    Designed mainly to generate scatterplot of performance metrics (e.g. auc-runtime).
    The function expects/demands a dataframe in which every row is a point to be plotted.
    Here statistics like the std must be provided in the dataframes. In other words they must be pre-computed.
    Here the hue and style column are just for visualization no aggregate statistic compututation is done on them.
    Labels the points with the text in "label_column" column.

    Parameters:
        ax (Axes): 
            Axes onto which draw the plot.
        
        df (pd.DataFrame): 
            Dataframe containing the data to plot.
        
        x_column (str): 
            Column plotted on x axis.
        
        y_column (str): 
            Column plotted on y axis.
        
        x_std_column(str | None, optional): 
            Name of the column containing the x std info.
            If None no x error bar is plotted.
        
        y_std_column (str | None, optional):
            Name of the column containing the x std info.
            If None no x error bar is plotted.
        
        hue_column (str | None, optional): 
            Name of the column to map as color. Can be None.

        style_column (str | None, optional):
            Name of the column to map to shapes. Can be None.

        map_style (dict | None, optional): 
            Map of shapes to use for the shape column.
        
        palette (dict | None, optional): 
            Dict of color mappings. If None the error bars (if any) will be gray.
        
        show_pareto_frontier (bool, optional):
            Show the pareto frontier obtained considering 
            x and y columns as objective to minimize.
            The frontier is computed over all points.
            The hue and style columns has no effect on this specification.
        
        label_column (str | None, optional): 
            Name of the column with point labels info. Can be None.

        error_bar_args (dict | None, optional):
            Dict unpackaged in the "errorbar" matplotlib function.
        
        sns_scatterplot_args (dict | None, optional):
            Dict unpackaged in the "scatterplot" seaborne function.

        sns_lineplot_frontier_args (dict | None, optional):
            Dict unpacked in the "lineplot" seaborne function used to plot the Pareto frontier.
            Ignored when "show_pareto_frontier" is False.
            
    Returns:
        Axes: The axes.
    '''
    cols = [x_column, y_column]

    for col in [label_column, hue_column, style_column]:
        cols = append_if_not_none(cols, col)

    check_presence_cols(df, cols)

    sns_scatterplot_args = ensure_or_create(sns_scatterplot_args, dict)
    error_bar_args = ensure_or_create(error_bar_args, dict)

    ax = sns.scatterplot(
        data=df, 
        x=x_column,
        y=y_column,
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
                    row[x_column],
                    row[y_column],
                    row[label_column]
                )
        )
        _ = adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    df_groups_err_bar = df.groupby(hue_column) if hue_column else [(None, df)]
    
    # the logic here is that we we want the error bar colored according to hue column
    # and error_bar takes vectors of positions ans values that are cross-linked by position
    # wehn no hue column is provided then we use the whole dataset in one pass to draw all error bars
    for hue_category, df_hue_category in df_groups_err_bar:
        color = "gray" \
            if palette is None or hue_category is None or hue_category not in palette.keys() \
            else palette[hue_category]

        if x_std_column:
            _ = ax.errorbar(
                x=df_hue_category[x_column],
                y=df_hue_category[y_column],
                xerr=df_hue_category[x_std_column],
                fmt="none",
                ecolor=color,
                **error_bar_args
            )
        
        if y_std_column:
            _ = ax.errorbar(
                x=df_hue_category[x_column],
                y=df_hue_category[y_column],
                yerr=df_hue_category[y_std_column],
                fmt="none",
                ecolor=color,
                **error_bar_args
            )

    if show_pareto_frontier:
        sns_lineplot_frontier_args = ensure_or_create(sns_lineplot_frontier_args, dict)
        pareto_mask = paretoset(df[[x_column, y_column]], sense=["min", "min"])
        pareto_subset = df.loc[pareto_mask, :]
        pareto_subset = pareto_subset.iloc[pareto_subset[x_column].argsort(), :]
        sns.lineplot(pareto_subset, x=x_column, y=y_column, ax=ax, **sns_lineplot_frontier_args)

    return ax



def draw_scatter_diagonal_plot(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    hue_column: str | None = None,
    style_column: str | None = None,
    palette: dict | None = None,
    map_style: dict | None = None,
    top_left_anno: str | None = None,
    bottom_right_anno: str | None = None,
    sns_scatterplot_args: dict | None = None
) -> Axes:
    '''
    Plot paired xy values in a scatter plot with a diagonal line representing equal values.

    Parameters:
        ax (Axes): 
            Axes onto which draw the plot.

        df (pd.DataFrame): 
            Dataframe containing the data to plot.

        x_column (str): 
            Name of the column to visualize on the x axis.

        y_column (str): 
            Name of the column to visualize on the y axis.

        hue_column (str | None, optional): 
            Name of the column mapped to hue.

        style_column (str | None, optional): 
            Name of the column mapped to shape.

        palette (dict | None, optional):  
            Palette of colors to use for the hue column.

        map_style (dict | None, optional): 
            sMap of shapes to use for the shape column.

        top_left_anno (str | None, optional): 
            Text annotation to report on the top left corner of the plot

        bottom_right_anno (str | None, optional):
            Text annotation to report on the down right corner of the plot.

        sns_scatterplot_args (dict, optional):
            Dict unpackaged in the "scatterplot" seaborne function.
    
    Returns:
        Axes: The axes.
    '''
    cols = [x_column, y_column]

    for col in [hue_column, style_column]:
        cols = append_if_not_none(cols, col)

    check_presence_cols(df, cols)

    sns_scatterplot_args = ensure_or_create(sns_scatterplot_args, dict)

    ax = sns.scatterplot(
        data=df, 
        x=x_column,
        y=y_column,
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
            fontsize=8
        )

    if bottom_right_anno is not None:
        ax.text(
            x=0.98, 
            y=0.02, 
            s=bottom_right_anno, 
            transform=ax.transAxes,
            ha='right', 
            va='bottom',
            fontsize=8
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