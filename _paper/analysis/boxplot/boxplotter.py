import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from typing import Literal
from matplotlib.axes import Axes
from statannotations.Annotator import Annotator
from metatab._paper.analysis.boxplot.pairs import extract_pairs
from metatab._paper.analysis.boxplot.types import simple_pairs, complex_pairs

from metatab._paper.analysis.boxplot.pvalues import (
    get_stat_arrays_from_pairs, 
    execute_test, 
    correct_pvalues
)

from metatab._paper.analysis.utils import (
    append_if_not_none,
    check_presence_cols,
    check_numeric_column,
    ensure_or_create,
    safe_update_dict
)




class BoxPLotter():
    '''
    Class to generate boxplots.

    Design:
        This class enables to quickly generate boxplots on ML perfomances data.
        In particular it enables to generate both simple and nested boxplots, i.e.
        with or without an "inner" hue mapping over high-level categories, 
        with a flexible mechanism for computing, correcting and plotting pvalues 
        for specific comparisons.

    Developer and User Note: 
        The class fails if x and hue columns contains container objects that 
        are not strings (tuples, lists, ...). This is currently not addressed or checked.
    '''
    def draw(
        self,
        ax: Axes,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        hue_column: str | None = None,
        paired_column: str | None = None,
        palette: None | dict = None,
        sns_boxplot_args: dict | None = None,
        test: None | Literal["Mann-Whitney", "t-test", "Wilcoxon"] = None,
        test_params: None | dict = None,
        pairs_to_annotate: Literal["inner_x", "all"] | simple_pairs | complex_pairs = "inner_x",
        pvalue_correction: None | Literal["inner_x", "all"] | simple_pairs | complex_pairs = None,
        pvalue_correction_method: Literal["bh", "by"] = "bh",
        alpha: float = 0.05,
        draw_pvalues: bool = True,
        pvalue_location: Literal["inside", "outside"] = "inside",
        hide_non_significant: bool = False,
        pvalue_thresholds: list[list[float, str]] = None
    ) -> tuple[Axes, pd.DataFrame | None]:
        '''
        Drawn boxplots. See class description for more info.

        Parameters:
            ax (Axes): Axes onto which draw the plot.

            df (pd.DataFrame): Data.

            x_column (str):
                String reporting the name of the column for which the main groups are formed.
                These are mapped to the x-axis.
            
            y_column (str): 
                String reporting the name of the column from which values the boxes are generated.
            
            hue_column (str | None, optional):
                String reporting the name of the column defining the subgroups and to mapped to hue.
                If None, the default, no column is mapped.

            paired_column (str | None, optional):
                String reporting the name of the column defining the pairing samples.
                The compared pairs are ordered based on this column values.
                If None, the default, no paired ordering is done.
            
            palette (dict | None, optional): 
                Palette of colors to use for the hue column.

            sns_boxplot_args (dict, optional):
                Dict unpackaged into the seaborn "boxplot" function. 
            
            test (None | Literal["Mann-Whitney", "t-test"], optional): 
                Statistical test to execute between the pairs to annotate.
                "mannwhitneyu", "ttest_ind" and "wilcoxon" scipy functions are used 
                for "Mann-Whitney", "t-test" and "Wilcoxon" options, respectively.
                The "Wilcoxon" option require that `paired_column` is set.
                If None, the default, no testing is perfomed.

            test_params (None | dict, optional):
                Dict of parameters passed to "ttest_ind", "mannwhitneyu" and "wilcoxon" scipy functions.
            
            pairs_to_annotate (Literal["inner_x", "all"] | simple_pairs | complex_pairs, optional):
                Specify between which boxes/groups compute and visualize the p-values.
                - "inner_x", the default, to compare all the inner x-categories levels 
                (so the tests are perfomed intra-x-category only).
                - "all", to compare all pairs against all the others (intra and inter x-categories).
                - simple_pairs | complex_pairs, to specify directly for which comparisons to compute pvalues.
                For the complex pairs one must specify before the x-category and then the inner level.
                In other words the order to follow is (x_column, hue_column).
            
            pvalue_correction (None | Literal["inner_x", "all"] | simple_pairs | complex_pairs, optional):
                How to perform multiple hypothesis correction:
                - None, the default, no correction is done.
                - "inner_x", the correction is done separately for each x-category.
                - "all", the correction is done for all comparisons.
                - simple_pairs | complex_pairs, the correction is done only on these pairs.
                For the complex pairs one must specify before the x-category and then the inner level.
                In other words the order to follow is (x_column, hue_column).
                    
            pvalue_correction_method (Literal["bh", "by"], optional): 
                Adjusting method to use:
                -"bh": Benjamini-Hochberg
                -"by": Benjamini-Yekutieli                
                Ignored when `pvalue_correction` is None.

            alpha (float, optional):
                Set the pvalue significance threshold. Defaults to 0.05.

            draw_pvalues (bool, optional):
                Wheter to draw pvalues on boxplot. 
                If False and "test" is set, the function returns the dataframe with test info without plotting.
            
            pvalue_location (Literal["inside", "outside"], optional):
                Whether to annotate pvalues and relative bars inside or outside the plot.
                Ignored when `test` is None or `draw_pvalues` if False.

            hide_non_significant (bool, optional):
                Whether to hide the non significant pvalues alpha-threhold-wise.
                Ignored when `test` is None or `draw_pvalues` if False.
            
            pvalue_thresholds (list[list[float, str]], optional):
                List of [number, string]. For the p_values < to {number} the string "< {string}" is used for annotations.
                When None, the default, is set to [[alpha, str(alpha)]].
                Ignored when `test` is None or `draw_pvalues` if False.

        Returns: 
            tuple[Axes,pd.DataFrame|None]: 
                A tuple with the plot and the dataframe collecting the test results.
                If no test is performed the second element is None.
        '''
        cols_to_check = append_if_not_none([y_column, x_column], hue_column)
        cols_to_check = append_if_not_none(cols_to_check, paired_column)
        grouping_cols = append_if_not_none([x_column], hue_column)
        
        check_presence_cols(df, cols_to_check)
        check_numeric_column(df, y_column)

        if test == "Wilcoxon" and paired_column is None:
            raise ValueError(
                "'Wilcoxon' test method requires the 'paired_column' to be specified."
            )

        sns_boxplot_args = ensure_or_create(sns_boxplot_args, dict)
        
        sns_boxplot_args = safe_update_dict(
            sns_boxplot_args,
            {"data": df, "x": x_column, "y": y_column, "hue": hue_column, "palette": palette, "ax": ax} 
        )

        ax = sns.boxplot(**sns_boxplot_args)
        df_tests = None

        if test:
            self._check_ambiguous_test_scenario(hue_column, pairs_to_annotate, pvalue_correction)
            test_params = ensure_or_create(test_params, dict)

            pairs = extract_pairs(df, x_column, hue_column, pairs_to_annotate) \
                if isinstance(pairs_to_annotate, str) \
                else pairs_to_annotate
            
            stat_arrays_pairs = get_stat_arrays_from_pairs(df, pairs, grouping_cols, y_column, paired_column)
            test_results = [execute_test(a, b, test, **test_params) for a, b in stat_arrays_pairs]      
            pvalues = np.array([test_result[0] for test_result in test_results])
            test_statistics = np.array([test_result[1] for test_result in test_results])

            corrected_pvalues, array_flag_corrections, array_group_correction = correct_pvalues(
                pvalues, 
                pairs, 
                pvalue_correction, 
                pvalue_correction_method
            )
            
            correction_strategy = pvalue_correction \
                if isinstance(pvalue_correction, str) or pvalue_correction is None \
                else "custom_pairs"
            
            correction_method = pvalue_correction_method if pvalue_correction else None
            effect_size_stats = self._compute_effect_size_stats(stat_arrays_pairs, paired_column)

            df_tests = self._build_df_tests(
                pairs=pairs, 
                pvalues=pvalues, 
                post_correction_pvalues=corrected_pvalues, 
                correction_strategy=correction_strategy,
                correction_flag=array_flag_corrections,
                correction_group=array_group_correction,
                correction_method=correction_method,
                test_statistics=test_statistics,
                dict_effect_size=effect_size_stats
            )
            
            if draw_pvalues:
                config_annotator = self._set_config_annotator(alpha, hide_non_significant, pvalue_thresholds)
                annotator = Annotator(pairs=pairs, **sns_boxplot_args,)
                ax, _ = annotator.configure(**config_annotator, loc=pvalue_location).set_pvalues_and_annotate(corrected_pvalues)
        
        return ax, df_tests



    @staticmethod
    def _check_ambiguous_test_scenario(
        hue_column: None | str,
        pairs_to_annotate: Literal["inner_x", "all"] | simple_pairs | complex_pairs,
        pvalue_correction: None | Literal["inner_x", "all"] | simple_pairs | complex_pairs,
    ) -> None:
        '''
        Checks the following incompatible or ambiguous scenarios 
        when statistical testing is desired:
        - hue_column not specified and pairs_to_annotate equal to "inner_x"
        - hue_column not specified and pvalue_correction equal to "inner_x"
        '''
        def is_inner_x(value) -> bool:
            return isinstance(value, str) and value == "inner_x"

        if hue_column is None:
            if is_inner_x(pairs_to_annotate):
                raise ValueError(
                    "When 'hue_column' is not specified, 'pairs_to_annotate' cannot be 'inner_x'."
                )  
            if is_inner_x(pvalue_correction):
                raise ValueError(
                    "When 'hue_column' is not specified, 'pvalue_correction' cannot be 'inner_x'."
                )


    @staticmethod
    def _set_config_annotator(
        alpha: float, 
        hide_non_significant: bool, 
        pvalue_thresholds: None | list[list[float, str]]
    ) -> dict:     
        pvalue_thresholds = [[alpha, str(alpha)]] if pvalue_thresholds is None else pvalue_thresholds
        return {
            "verbose": 0, 
            "text_format": "simple", 
            "alpha": alpha,
            "hide_non_significant": hide_non_significant,
            "pvalue_thresholds": pvalue_thresholds
        }
    

    @staticmethod
    def _compute_effect_size_stats(
        arrays: list[tuple[np.ndarray, np.ndarray]], 
        paired_column: str | None
    ) -> dict[str, list]:
        '''
        Compute statistics informing about the effect size and direction of comparison.
        In detail, computes difference between medians and means of the array, 
        plus mean and median of paired differences + ratio of 1 > 2 in paired scenario.        
        '''
        results = defaultdict(list)
        for array_1, array_2 in arrays:
            results["diff_mean_12"].append(np.nanmean(array_1) - np.nanmean(array_2))
            results["diff_median_12"].append(np.nanmedian(array_1) - np.nanmedian(array_2))
            if paired_column:
                diff = array_1 - array_2
                effective_size = (~(np.isnan(array_1) | np.isnan(array_2))).sum()
                count_1_over_2 = (array_1 > array_2).sum()
                count_2_over_1 = (array_2 > array_1).sum()
                results["mean_diff_12"].append(np.nanmean(diff))
                results["median_diff_12"].append(np.nanmedian(diff))
                results["count_1_over_2"].append(count_1_over_2)
                results["count_2_over_1"].append(count_2_over_1)
                results["fraction_1_over_2"].append(count_1_over_2/ effective_size)
                results["fraction_2_over_1"].append(count_2_over_1 / effective_size)
        return results


    @staticmethod
    def _build_df_tests(
        pairs: simple_pairs | complex_pairs, 
        pvalues: np.ndarray,
        post_correction_pvalues: np.ndarray,
        correction_flag: np.ndarray,
        correction_group: np.ndarray,
        correction_strategy: str | None,
        correction_method: str | None,
        test_statistics: np.ndarray,
        dict_effect_size: dict[str, list]
    ) -> pd.DataFrame:
        '''Builds and returns the dataframe with the pvalue information'''
        first_elements = [pair[0] for pair in pairs]
        second_elemets = [pair[1] for pair in pairs]
        return pd.DataFrame({
            "group1": first_elements,
            "group2": second_elemets,
            "pvalues": pvalues,
            "corrected_pvalues": post_correction_pvalues,
            "correction_flag": correction_flag,
            "correction_group": correction_group,
            "correction_strategy": correction_strategy,
            "correction_method": correction_method,
            "test_statistic": test_statistics,
            **dict_effect_size
        })