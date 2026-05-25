import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from typing import Literal
from matplotlib.axes import Axes
from matplotlib import lines as mlines
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
    The function set x_left/y_bottom and x_right/y_top to the min and max values found in xy
    to assure a correct interpreation of paired values.

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
    min_value = min(df[x_column].min(), df[y_column].min())
    max_value = max(df[x_column].max(), df[y_column].max())
    ax.set_xlim(right=max_value, left=min_value)
    ax.set_ylim(top=max_value, bottom=min_value)

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



def draw_stacked_bar_improvability(
    ax: Axes,
    df: pd.DataFrame,
    x_column: str,
    hue_column: str,
    y_column: str,
    paired_column: str,
    type_improvability: Literal["gap", "tabarena"],
    hline_x_hue_category: tuple[str, str] | None,
    hline_kwargs: None | dict = None,
    palette: dict | None = None,
    p_values=None,
    pvalue_alpha: float = 0.05
) -> tuple[pd.DataFrame, Axes]:
    """
    Stacked overlapping bar chart of metric improvability over hue categories.
    This implementation assumes that improvability is computed over AUC values.

    Parameters:
        ax (Axes): 
            Axes onto which draw the plot.
        
        df (pd.DataFrame):
            Long-format DataFrame containing all relevant columns.

        x_column (str):
            Column defining the categories shown on the x-axis.

        hue_column (str):
            Column defining the groups shown as stacked bars.

        y_column (str):
            Column containing the performance metric.

        paired_column (str):
            Column identifying the paired observations used to compute improvability
            Improvability is averaged over unique values of this column.

        type_improvability (Literal["gap", "tabarena"]):
            Type of improvability metric:
            - "gap": mean across paired obs of the differences max(auc) minus model auc within paired obs.
            - "tabarena": quantify the mean across paired obs of how much error you can recover by switching to the best method.

        palette (dict | None, optional):
            Colors for each hue level in the order they appear in hue_column.
            If None, uses matplotlib's default color cycle.

        hline_kwargs (dict | None, optional):
            Keyword arguments forwarded to ax.axhline. Recognized extra keys:
                - "label" (str): label shown in the legend for the hline.
            If None, defaults to {"linestyle": "--", "linewidth": 2.5, "color": "black"}.

        p_values (dict, optional):
            Significance annotations. Keys are (x_value, hue_a, hue_b) tuples,
            values are p-values. Pairs with p < 0.05 are annotated with circled
            numbers above each bar. Lookup is symmetric: (clf, a, b) and (clf, b, a)
            are treated as the same comparison.

        pvalue_alpha (float, optional):
            Significance thresold used for pvalues displaying.
            Ignored when p_values is None.

    Returns:
        tuple[Axes,pd.Series]: 
        A tuple of the improvability values and the plotted axes.
    """
   # Defaults
    _hline_kwargs = {"linestyle": "--", "linewidth": 2.5, "color": "black"}
    if hline_kwargs is not None:
        _hline_kwargs.update(hline_kwargs)
    hline_label = _hline_kwargs.pop("label", None)
    hline_x, hline_hue = hline_x_hue_category if hline_x_hue_category is not None else (None, None)
    
    # Hues and colors 
    hues = [
        h for h in df[hue_column].unique()
        if hline_hue is None or h != hline_hue
    ]

    if palette is None:
        palette = {h: f"C{i}" for i, h in enumerate(hues)}
    
    hue_colors = [palette[h] for h in hues]

    # Build wide table
    df = df.copy()
    df["_key"] = df[hue_column].astype(str) + "--" + df[x_column].astype(str)
    df_wide = df[["_key", paired_column, y_column]].pivot(columns="_key", index=paired_column, values=y_column)

    # Check hline category in df_wide
    if hline_x_hue_category is not None:
        hline_key = f"{hline_hue}--{hline_x}"
        if hline_key not in df_wide.columns:
            raise ValueError(
                f"hline_x_hue_category {hline_x_hue_category!r} not found in data."
            )

    # Improvability computation
    if type_improvability == "tabarena":
        full_imp = df_wide.apply(lambda row: (row.max() - row) / (1 - row) * 100, axis=1).mean(axis=0)
    else:
        full_imp = df_wide.apply(lambda row: row.max() - row, axis=1).mean(axis=0)

    # hline value management
    if hline_x_hue_category is not None:
        hline_value = full_imp[hline_key]
        imp = full_imp.drop(hline_key)
    else:
        hline_value = None
        imp = full_imp

    # imp_matrix: x_values x hues (excluding hline category)
    x_values = [
        v for v in df[x_column].unique()
        if hline_x_hue_category is None or v != hline_x
    ]
    
    imp_matrix = pd.DataFrame({
        x: {h: imp[f"{h}--{x}"] for h in hues}
        for x in x_values
    }).T.reindex(columns=hues)

    #  Order x by min improvability (lowest = best = leftmost) 
    x_order = imp_matrix.min(axis=1).sort_values().index.tolist()
    imp_matrix = imp_matrix.loc[x_order]

    #  Sort hues per x: worst first (widest bar behind), best last (narrowest on top) 
    rank_matrix = np.argsort(imp_matrix.values, axis=1)[:, ::-1]
    sorted_imp = np.take_along_axis(imp_matrix.values, rank_matrix, axis=1)
    sorted_colors = np.array(hue_colors)[rank_matrix]

    n_hues = len(hues)
    widths = np.linspace(0.6, 0.2, n_hues)
    x = np.arange(len(x_order))

    # Bars
    for level in range(n_hues):
        ax.bar(
            x, sorted_imp[:, level], 
            bottom=0,
            color=sorted_colors[:, level],
            width=widths[level],
            zorder=level,
        )

    # Significance annotations 
    if p_values is not None:
        # get all pairwise comparisons
        comparisons = [(a, b) for i, a in enumerate(hues) for b in hues[i + 1:]]
        # associate a symbol to each comparison 
        symbols = [chr(0x2460 + i) for i in range(len(comparisons))]

        for xi, x_val in enumerate(x_order):
            y_top = imp_matrix.loc[x_val].max()
            parts = []
            for (h_a, h_b), sym in zip(comparisons, symbols):
                p = (
                    p_values.get((x_val, h_a, h_b)) or
                    p_values.get((x_val, h_b, h_a)) or
                    1.0 # fallback to 1 when not found
                )
                parts.append(sym if p <= pvalue_alpha else " ")
            ax.text(
                x=xi, 
                y=y_top + y_top * 0.01,
                s="".join(parts),
                ha="center", 
                va="bottom", 
                fontsize=12
            )

        legend_text = "\n".join(
            f"{sym} {a} vs {b}" for sym, (a, b) in zip(symbols, comparisons)
        ) + f"\n(p <= {pvalue_alpha})"

        ax.text(
            x=1.01, 
            y=0.99, 
            s=legend_text,
            transform=ax.transAxes, 
            fontsize=10,
            va="top", 
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8)
        )

    # Hline 
    if hline_value is not None:
        ax.axhline(y=hline_value, **_hline_kwargs)

    # Legend 
    handles = [mpatches.Patch(color=c, label=h) for c, h in zip(hue_colors, hues)]
    if hline_value is not None and hline_label is not None:
        handles.append(
            mlines.Line2D(
                [], [],
                color=_hline_kwargs.get("color", "black"),
                linestyle=_hline_kwargs.get("linestyle", "--"),
                linewidth=_hline_kwargs.get("linewidth", 2.5),
                label=hline_label,
            )
        )
    ax.legend(handles=handles, title=hue_column, fontsize=10, title_fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(x_order, rotation=45, ha="right")
    ax.set_ylabel("Improvability")
    return ax, full_imp