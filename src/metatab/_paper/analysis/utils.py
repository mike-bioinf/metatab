from __future__ import annotations

import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from typing import Literal, Any, Callable, TYPE_CHECKING
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend import Legend

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from autorank._util import RankResult



def enlist(x: Any | list, none_as_is = False) -> list:
    '''
    Put into a list x if not already a list.
    If x is None one can decide to not enlist it.
    '''
    if isinstance(x, list):
        return x
    elif x is None and none_as_is:
        return None
    else:
        return [x]


def check_presence_cols(df: pd.DataFrame, cols: str | list[str]) -> None:
    '''
    Check the presence of multiple columns in a dataframe.
    The utility assumes that the columns are strings.
    '''
    for col in enlist(cols):
        check_presence_col(df, col)


def check_presence_col(df: pd.DataFrame, col: str) -> None:
    '''
    Check the presence of a single column in df.
    The utility assumes that the df columns are strings.
    '''
    if col not in df.columns:
        raise ValueError(f"'{col}' column is not found in df.")


def check_numeric_column(df: pd.DataFrame, col: str):
    '''Check the numeric nature of the column'''
    current_dtype = df.dtypes[col]
    if not pd.api.types.is_numeric_dtype(current_dtype):
        raise TypeError(
            f"column '{col}' is expected to be numeric, instead it is of type '{str(current_dtype)}'."
        )


def append_if_not_none(l: list, obj: Any) -> list:
    '''
    Append the object to the list if not None.
    Returns the list.
    '''
    if not isinstance(l, list):
        raise TypeError("l msut be a list object.")    
    if obj is not None:
        l.append(obj)
    return l


def ensure_or_create(obj: Any, constructor: Callable[[], Any]) -> Any:
    """
    Return `obj` if it evaluates to True, 
    otherwise create and return a new instance by calling `constructor`.

    This is useful for safely initializing optional arguments, e.g.:
        my_dict = ensure_or_create(existing_dict, dict)

    Parameters:
        obj (Any):
            The object to check.
        constructor (Callable[[], Any]):
            A zero-argument callable used to create a new object if `obj` is falsy.

    Returns:
        Any: The original object if truthy, otherwise a newly constructed one.
    """
    return obj if obj else constructor()


def _compute_normalized_score(s: pd.Series) -> pd.Series:
    '''Computes the normalized scores following TabRepo'''
    median_s = s.median()
    return ((s - median_s) / (s.max() - median_s)).apply(lambda v: max(0, v))


def compute_aggregate_statistics(
    d_wide: pd.DataFrame,
    skip_secondary_stats: bool = True
) -> pd.DataFrame:
    '''
    Compute aggregate statistics on a wide dataframe (dataset x classifier).
    The function expects datasets in index and classifiers in columns.

    In detail computes:
        -average_rank --> average of per-row ranks across rows.
        -std_rank --> std in ranks across rows.
        -wins --> Absolute number of times each classifier is the best per row across rows.
        -wins_ratio --> express #wins over the total number of rows.
        -average_regret --> average regret (distance from the best per row) across rows.
        -std_regret --> std of regret across rows.
        -improvability --> improvability metric defined following TabArena.
        -normalized_score --> score defined in TabRepo.
        -quartiles --> median, q25 and q75 values.

    Notes: 
    1) Ranking and winning metrics use the average method to resolve ties.
    This means that wins do not include draws. 
    2) Ranking, winning, improvability, regret and normalized scores 
    metrics are computed across classifiers and then averaged across datasets.
    Quartiles are instead compute directly across datasets by classifier.
    
    Parameters:
        d_wide (pd.DataFrame): 
            Wide DataFrame with datasets in index and classifier in columns.
        skip_secondary_stats (bool, optional):
            Whether to skip regret and normalized score statistics computation.

    Returns:
        pd.DataFrame: DataFrame with all the aggregate statistics. 
    '''
    d_rank = d_wide.rank(axis=1, ascending=False, method="average") # ties are resolved as averages
    d_improvabity = d_wide.apply(lambda row: ((row.max() - row) / (1 - row) * 100), axis=1)
    wins = (d_rank == 1).sum(axis=0)

    stats = pd.DataFrame({
        # Rank statistics
        "average_rank": d_rank.mean(axis=0),
        "std_rank": d_rank.std(axis=0),
        # Win statistics
        "wins": wins,
        "wins_ratio": wins / d_wide.shape[0],
        # Improvability
        "average_improvability": d_improvabity.mean(axis=0),
        "std_improvability": d_improvabity.std(axis=0),
        # Raw score statistics
        "median_score": d_wide.median(axis=0),
        "score_q25": d_wide.quantile(0.25, axis=0),
        "score_q75": d_wide.quantile(0.75, axis=0),
    })

    if not skip_secondary_stats:
        d_regret = d_wide.apply(lambda row: row.max() - row, axis=1)
        stats_secondary = pd.DataFrame({
            # Regret statistics
            "average_regret": d_regret.mean(axis=0),
            "std_regret": d_regret.std(axis=0),
            # Normalized scores
            "average_normalized_score": d_wide.apply(lambda row: _compute_normalized_score(row), axis=1).mean(axis=0),
        })
        stats = pd.concat([stats, stats_secondary], axis=1)

    return stats


def save_autorank_results_to_excel(autorank_result: RankResult, path: str | Path) -> None:
    '''
    Save the autorank results in a excel file.

    Parameters:
        autorank_result (RankResult): 
            Autorank main object derived from "autorank" function.
        path (str | Path):
            Filepath of the resulting excel file.
    '''
    excel_writer = pd.ExcelWriter(path) 
    autorank_test_stats = pd.DataFrame(
        [[autorank_result.omnibus, autorank_result.pvalue, autorank_result.cd]],
        columns=["omnibus_test", "omnibuos_pvalue", "nemenyi_cd"]
    )
    autorank_methods_stats = autorank_result.rankdf.drop(columns=["magnitude", "magnitude_above"])
    autorank_methods_stats.to_excel(excel_writer)
    autorank_test_stats.to_excel(excel_writer, sheet_name="Sheet2", index=False)
    excel_writer.close()


def get_legend_cross_classifiers(
    palette_single_classifiers: dict,
    palette_cross_classifiers: dict,
    shape_map: dict,
    fallback_marker: str = "o",
) -> list:
    '''
    Build grouped legend handles for:
    1) single classifiers (color-coded circles)
    2) regimes (shape-coded, black)
    3) cross classifiers (color + inferred shape)
    
    Returns list of handles that can be passed to "ax.legend()" calls.
    '''
    # --- section 1: single classifiers ---
    single_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=color,
            linestyle="none",
            markersize=7,
            label=label,
        )
        for label, color in palette_single_classifiers.items()
    ]

    # --- section 2: regimes (black + shapes) ---
    regime_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            color="black",
            linestyle="none",
            markersize=7,
            label=label,
        )
        for label, marker in shape_map.items()
    ]

    # --- helper: infer shape for cross-classifier ---
    def get_shape_for_cross(label: str) -> str:
        for mode, marker in shape_map.items():
            if label.startswith(mode):
                return marker
        return fallback_marker

    # --- section 3: cross classifiers ---
    cross_handles = [
        Line2D(
            [0], [0],
            marker=get_shape_for_cross(label),
            color=color,
            linestyle="none",
            markersize=7,
            label=label,
        )
        for label, color in palette_cross_classifiers.items()
    ]

    spacer = Patch(visible=False, label="")

    return (
        [Patch(visible=False, label="Single Classifiers")]
        + single_handles
        + [spacer]
        + [Patch(visible=False, label="Regime")]
        + regime_handles
        + [spacer]
        + [Patch(visible=False, label="Cross Classifiers")]
        + cross_handles
    )


def _aggregate_df_search(
    df_search: pd.DataFrame, 
    groupby_column: str = "search_iter",
    loss_column: str = "loss",
    remove_groupby_column: bool = False
) -> pd.DataFrame:
    '''
    Abstract the logic to aggregate the df search.
    Apply mean aggregation on the loss column and first aggregation on the others.
    Returns the aggregated dataframe.
    '''
    agg_dict = {}
    for col in df_search.columns:
        if col == loss_column:
            agg_dict[col] = ["mean", "std"]
        else:
            agg_dict[col] = "first"

    del agg_dict[groupby_column]
    df_search_agg = df_search.groupby(groupby_column).agg(agg_dict).reset_index()

    # flatten column multiindex
    df_search_agg.columns = [
        loss_column if c == (loss_column, "mean")
        else f"std_{loss_column}" if c == (loss_column, "std")
        else c[0]
        for c in df_search_agg.columns
    ]

    if remove_groupby_column:
        del df_search_agg[groupby_column]

    return df_search_agg


def load_search_data(path: str | Path) -> pd.DataFrame:
    '''
    Load the search data and automatically aggregate cv losses across cv iterations. 
    Wants in input the path of the estimator folder containing files for each dataset.
    Returns the aggregated search data.
    '''
    path = Path(path) if isinstance(path, str) else path
    data = []
    for f in path.iterdir():
        d = pd.read_csv(f, sep="\t")
        del d["fold"]
        del d["repeat"]
        d = _aggregate_df_search(
            d,
            groupby_column="search_iter",
            remove_groupby_column=True
        )
        d["dataset"] = f.stem
        data.append(d)
    return pd.concat(data, axis=0, ignore_index=True)


def load_complete_metadata(
    path_actual_pred_losses: str | Path, 
    path_search_data: str | Path
) -> pd.DataFrame:
    '''
    Function that help to build the complete metadata for an estimator, 
    i.e the dataset with true raw and z-normalized losses, predicted z-normalized losses and hps.

    IMPORTANT: the function relies on the assumption that the order of hps configurations
    for a (dataset, classifier) combination isn the same between the two sources.
    This is not enforced or checked therefore be attentive.

    Parameters:
        path_actual_pred_losses (str | Path):
            Path of the estimator file containing info for all datasets (es. ...ablation/rf_1500/catboost)
        path_search_data (str | Path):
            Path to the estimator folder containing files for each dataset (es. ...search_data/catboost).
            Require the folder given in input to 'load_search_data'.

    Returns:
        pd.DataFrame: The complete metadata for the target estimator.
    '''
    df_sd = load_search_data(path_search_data)
    df_losses = pd.read_csv(path_actual_pred_losses, sep="\t")
    
    assert df_sd.shape[0] == df_losses.shape[0], "dataframes number of rows does not match"
    assert set(df_sd["dataset"]) == set(df_losses["dataset"]), "datasets does not match between search data and actual_pred data"
    
    # we assume same row order inside dataset block
    df_sd["row_in_dataset"] = df_sd.groupby("dataset").cumcount()
    df_losses["row_in_dataset"] = df_losses.groupby("dataset").cumcount()

    df_merged = df_sd.merge(
        df_losses,
        on=["dataset", "row_in_dataset"]
    ).drop(columns="row_in_dataset")

    return df_merged


def get_surrogate_feature_importance_scores(surrogate_pipe: Pipeline) -> pd.Series:
    '''
    Get the feature importance score from the surrogate.
    It handles the one hot encoded scores by summing the categories scores.
    '''
    surrogate_model = surrogate_pipe[-1]

    scores = pd.Series(
        surrogate_model.forest_.feature_importances_,
        index=surrogate_model.forest_.feature_names_in_,
        name="importance",
    )

    column_transformer: ColumnTransformer = surrogate_pipe.named_steps["columntransformer"]

    if "onehot" not in column_transformer.named_transformers_:
        return scores.sort_values(ascending=False)

    ohe: OneHotEncoder = column_transformer.named_transformers_["onehot"]
    mapping = {}

    # Build mapping from actual encoder output names to original features.
    for feature_in, categories in zip(
        ohe.feature_names_in_,
        ohe.categories_
    ):
        for category in categories:
            encoded_name = f"{feature_in}_{category}"
            if encoded_name in scores.index:
                mapping[encoded_name] = feature_in

    return (
        scores.rename(index=lambda feature: mapping.get(feature, feature))
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )


def save_plot(fig, filepath: str | Path) -> None:
    '''
    Save figure in png and svg.
    The function does not require the extension in filepath.
    '''
    filepath = str(filepath) if isinstance(filepath, Path) else filepath
    fig.savefig(f"{filepath}.svg", bbox_inches="tight")
    fig.savefig(f"{filepath}.png", bbox_inches="tight", dpi=600)


def save_boxplot_df_tests_to_excel(
    df_tests: pd.DataFrame, 
    filepath: str | Path,
    format_numeric_columns: bool = False,
) -> None:
    '''
    Save in excel the df_tests returned by BoxPlotter.
    Want the filepath with the extension.
    '''
    df_tests = df_tests.copy()

    columns_to_drop = [
        'correction_flag',
        'correction_group',
        'correction_strategy',
        'mean_victory_1',
        'mean_victory_2'
    ]

    if format_numeric_columns:
        pvalue_cols = ["pvalue", "corrected_pvalue"]
        non_format_cols = ["test_statistic", "count_2_over_1", "count_1_over_2"]
        
        for col in pvalue_cols:
            df_tests[col] = df_tests[col].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")

        numeric_cols = [
            col for col in df_tests.columns
            if col not in pvalue_cols + non_format_cols and is_numeric_dtype(df_tests[col])
        ]

        for col in numeric_cols:
            df_tests[col] = df_tests[col].map("{:.2f}".format)

    df_tests.drop(columns=columns_to_drop).to_excel(filepath, index=False)


def remove_unused_regime_classifier_categories(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove the unused categories in 'Regime' and 'Classifier' columns.'''
    if is_categorical_dtype(df["Regime"]):
        df["Regime"] = df["Regime"].cat.remove_unused_categories()
    if is_categorical_dtype(df["Classifier"]):
        df["Classifier"] = df["Classifier"].cat.remove_unused_categories()
    return df