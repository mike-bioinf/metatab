import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from adjustText import adjust_text
from _paper.analysis.utils import check_presence_cols



def draw_scatterplot_auc_runtime(
    ax: Axes, 
    df: pd.DataFrame,
    auc_column: str,
    runtime_column: str,
    auc_std_column: str | None = None,
    runtime_std_column: str | None = None,
    hue_column: str | None = None,
    palette: None | dict = None,
    space_label_column: str | None = None,
    error_bar_args: dict = {},
    sns_scatterplot_args: dict = {}
) -> None:
    '''
    Plot the auc-runtime scatterplot on the input Axes.
    Labels the points with the text in "space_label_column" column.

    Parameters:
        ax (Axes): Axes onto which draw the plot.
        
        df (pd.DataFrame): Dataframe containing the data to plot.
        
        auc_column (str): Name of the column containing the auc info.
        
        runtime_column (str): Name of the column containing the runtime info.
        
        auc_std_column(str | None, optional): 
            Name of the column containing the auc std info.
            If None no auc error bar is plotted.
        
        runtime_std_column (str | None, optional):
            Name of the column containing the runtime std info.
            If None no runtime error bar is plotted.

        hue_column (str | None, optional): 
            Name of the column to map as color. Can be None.
        
        palette (dict | None, optional): 
            Dict of color mappings. If None the error bars (if any) will be gray.
        
        space_label_column (str | None, optional): 
            Name of the column with the space configuration labels info. Can be None.
        
        error_bar_args (dict, optional):
            Dict unpackaged in the "errorbar" matplotlib function.
        
        sns_scatterplot_args (dict, optional):
            Dict unpackaged in the "scatterplot" seaborne function.
    '''
    cols = [auc_column, runtime_column]
    cols = cols + [space_label_column] if space_label_column is not None else cols
    cols = cols + [hue_column] if hue_column is not None else cols

    check_presence_cols(df, cols)
    
    ax = sns.scatterplot(
        data=df, 
        x=auc_column,
        y=runtime_column,
        hue=hue_column if hue_column is not None else None,
        palette=palette,
        ax=ax,
        **sns_scatterplot_args
    )

    if space_label_column:
        texts = []
        for _, row in df.iterrows():
            texts.append(
                ax.text(
                    row[auc_column],
                    row[runtime_column],
                    row[space_label_column]
                )
        )
        _ = adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))


    if hue_column is not None:
        for hue_category, df_hue_category in df.groupby(hue_column):
            color = "gray" \
                if palette is None or hue_category not in palette.keys() \
                else palette[hue_category]

            if auc_std_column:
                _ = ax.errorbar(
                    x=df_hue_category[auc_column],
                    y=df_hue_category[runtime_column],
                    xerr=df_hue_category[auc_std_column],
                    fmt="none",
                    ecolor=color,
                    **error_bar_args
                )
            
            if runtime_std_column:
                _ = ax.errorbar(
                    x=df_hue_category[auc_column],
                    y=df_hue_category[runtime_column],
                    yerr=df_hue_category[runtime_std_column],
                    fmt="none",
                    ecolor=color,
                    **error_bar_args
                )
