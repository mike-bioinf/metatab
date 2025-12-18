import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from typing import Literal, Iterable, Any
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from warnings import warn
from metatab.metatab_utils.prediction.metrics import compute_metrics

from metatab.metatab_utils.prediction.constants import (
    PERFORMANCE_METRICS,
    MUST_COLUMNS_TO_PARSE,
    OPTIONAL_COLUMNS_TO_PARSE,
    MANDATORY_COLUMNS,
    EXTRACTED_COLUMNS
)

from metatab.metatab_utils.prediction.utils import (
    wrap_into_list, 
    to_list_of_numpy_arrays, 
    are_same_length
)

from metatab.metatab_utils.prediction.parser import (
    save_ndarray_to_str,
    save_str_to_ndarray
)




class PredictionDataframe():
    '''
    Class to manage the I/O of classification inference data.

    The prediction dataframe has specific columns collecting specific info:
    "dataset", "classification_setting", "classes", "classes_counts", "test_labels", 
    "pred_labels" and "pred_proba".

    pd.NA/np.nan/None values are allowed only in "pred_labels" and "pred_proba" columns.

    As general design one can access and modify the underlying pandas DataFrame, 
    with the exception of these columns.
    '''
    def __init__(self):
        self.df: pd.DataFrame = None
        self.must_columns = MANDATORY_COLUMNS
        self.must_columns_to_parse = MUST_COLUMNS_TO_PARSE
        self.optional_columns_to_parse = OPTIONAL_COLUMNS_TO_PARSE
        self.metrics_columns = PERFORMANCE_METRICS
        self.all_columns_to_parse = (
            MUST_COLUMNS_TO_PARSE + 
            OPTIONAL_COLUMNS_TO_PARSE +
            PERFORMANCE_METRICS
        )
        self.has_recovered = None


    def __str__(self):
        return self.df.__str__() if self.df is not None else self


    def __repr__(self):
        return self.df.__repr__() if self.df is not None else self


    def build_from_data(
        self,
        dataset: str | list[str] | pd.Series,
        y_train: np.ndarray | pd.Series | list[pd.Series | np.ndarray],
        y_test: np.ndarray | pd.Series | list[pd.Series | np.ndarray],
        pred_proba: np.ndarray | list[np.ndarray],
        save_path: str | Path | None = None,
        **add_columns
    ) -> "PredictionDataframe":
        '''
        Build the PredictionDataFrame from data. 
        It presents a recover strategy that enables to save 
        the input data as a dict serialized in a pickle file,
        if an exception is raised during the process (see 'save_path' param).

        Parameters:
            dataset (str | list[str] | pd.Series): Dataset name/s.
            y_train (np.ndarray | pd.Series | list[pd.Series | np.ndarray]): Training labels.
            y_test (np.ndarray | pd.Series | list[pd.Series | np.ndarray]): Test labels.
            pred_proba (np.ndarray | list[np.ndarray]): Predicted probabilities.
            save_path (str | Path | None): 
                Path to the folder where the pickle file is saved.
                The file name is automatically created using the "partial_data__" prefix + time stamp.
                If None, the default, the recover strategy is not used.
            **add_columns: 
                Additional columns to add to the dataframe.
                The keys are used as column names and the values as column values.

        Returns:
            Self, the dataframe is stored in self.df.
        '''
        partial_data = {
            "dataset": dataset,
            "y_train": y_train,
            "y_test": y_test,
            "pred_proba": pred_proba,
            "add_columns": add_columns
        }

        try:
            for key in add_columns.keys():
                if key in self.must_columns:
                    raise KeyError(
                        f"Is not possible to add columns with one of the following names: {self.must_columns}"
                    )
            
            dataset, pred_proba, y_train, y_test = self._adapt_data(
                dataset, 
                pred_proba, 
                y_train, 
                y_test,
                copy=True
            )

            self._check_adapted_data(dataset, pred_proba, y_train, y_test)

            (
                classes, 
                classes_counts, 
                classification_setting, 
                pred_labels
            ) = self._extract_info_from_data(pred_proba, y_train)

            df = {
                "dataset": dataset,
                "classification_setting": classification_setting,
                "classes": classes,
                "classes_counts": classes_counts,
                "test_labels": y_test,
                "pred_labels": pred_labels,
                "pred_proba": pred_proba
            }

            df.update(**add_columns)
            self.df = pd.DataFrame(df)
            self.has_recovered = False
            return self

        except Exception as e:
            if save_path:
                dump_file = self._save_partial_data(partial_data, save_path)
                self.has_recovered = True
                print(f"Caught Exception in the PredictionDataframe building process: {e}. \nRecovering.", file=sys.stderr)
                print(f"Partial data saved to: {dump_file}", file=sys.stderr)
            raise
            


    def add_rows(
        self, 
        rows: dict | list[dict], 
        compute_metrics: bool = False,
        multiclass: Literal["ovr", "average"] | None = None, 
        average_strategy: Literal["micro", "macro", "weighted"] | None = None
    ) -> "PredictionDataframe":
        '''
        Adds rows to the underlying dataframe.
        If the dataframe is missing then it builds it.

        Parameters:
            rows (dict | list[dict]):
                Dictionaries to add as new rows. 
                They must have some mandatory keys.
            compute_metrics (bool, optional):
                Whether to compute the performance metrics on the new rows.
            multiclass (Literal["ovr", "average"] | None, optional):
                See "compute_metrics" method doc. Ignored when "compute_metrics" is False.
            average_strategy (Literal["micro", "macro", "weighted"] | None, optional):
                See "compute_metrics" method doc. Ignored when "compute_metrics" is False.

        Returns:
            PredictionDataframe: self.
        '''        
        if compute_metrics and (multiclass is None or average_strategy is None):
            raise ValueError("To compute metrics both 'multiclass' and 'average_strategy' must be specified.")

        rows = wrap_into_list(rows)

        for row in rows:
            # check type
            if not isinstance(row, dict):
                raise TypeError("Found non-dict objects in rows.")
            
            # check mandatory keys
            for must_column in ["dataset", "y_test", "y_train", "pred_proba"]:
                if must_column not in row.keys():
                    raise KeyError(
                        f"The following mandatory key is missing at least in one row-dict: {must_column}"
                    )
            
            # check conflicting keys
            conflicting_keys = EXTRACTED_COLUMNS + self.metrics_columns \
                if compute_metrics \
                else EXTRACTED_COLUMNS
            
            for conflicting_key in conflicting_keys:
                if conflicting_key in row.keys():
                    raise KeyError(
                        f"The following row-dict key causes conflicts: {conflicting_key}"
                    )
    
        # merge multiple dicts into a single extended one
        if len(rows) == 1:
            single_dict = deepcopy(rows[0])
        else:
            single_dict = defaultdict(list)
            for row in rows:
                for k, v in row.items():
                    single_dict[k].append(v)

        dataset, pred_proba, y_train, y_test = self._adapt_data(
            single_dict["dataset"],
            single_dict["pred_proba"],
            single_dict["y_train"],
            single_dict["y_test"],
            copy=True
        )

        self._check_adapted_data(dataset, pred_proba, y_train, y_test)
        
        (
            classes, 
            classes_counts, 
            classification_setting, 
            pred_labels
        ) = self._extract_info_from_data(pred_proba, y_train)

        single_dict["dataset"] = dataset
        single_dict["classification_setting"] = classification_setting
        single_dict["classes"] = classes
        single_dict["classes_counts"] = classes_counts
        single_dict["test_labels"] = y_test
        single_dict["pred_proba"] = pred_proba
        single_dict["pred_labels"] = pred_labels
        
        # y_train is not part of the output
        del single_dict["y_train"]
        # y_test must be nominated as test_labels for uniformity with build methods
        del single_dict["y_test"]

        df_to_add = pd.DataFrame(single_dict)

        if compute_metrics:
            df_to_add = self._compute_metrics(df_to_add, multiclass, average_strategy)

        self.df = pd.concat([self.df, df_to_add], axis=0, ignore_index=True)
        return self



    @staticmethod
    def _adapt_data(
        dataset: str | list[str] | pd.Series,
        pred_proba: np.ndarray | list[np.ndarray],
        y_train: np.ndarray | pd.Series | list[np.ndarray | pd.Series],
        y_test: np.ndarray | pd.Series | list[np.ndarray | pd.Series],
        copy: bool = False
    ) -> list[list, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        '''
        Adapt and return the data in the input order.
        The adaptation is done on a deepcopy when `copy` is True.
        '''
        dataset, pred_proba, y_train, y_test = wrap_into_list(
            dataset, 
            pred_proba,
            y_train, 
            y_test
        )
        
        y_train = deepcopy(y_train) if copy else y_train
        y_test = deepcopy(y_test) if copy else y_test

        y_train = to_list_of_numpy_arrays(y_train)
        y_test = to_list_of_numpy_arrays(y_test)
        
        return dataset, pred_proba, y_train, y_test



    def _check_adapted_data(
        self,
        dataset: list[str|pd.Series],
        pred_proba: Iterable[np.ndarray], 
        y_train: Iterable[np.ndarray], 
        y_test: Iterable[np.ndarray]
    ) -> None:
        '''Check on the adapted data used in the 'build_from_data' and 'add_rows' methods.'''
        if not are_same_length(dataset, pred_proba, y_train, y_test):
            raise ValueError((
                "The input iterables have not the same length." 
                " Note that 'scalar' inputs are internally converted to iterables."
            ))
        self._check_array_dims(pred_proba, 2, "pred_proba", allow_na=True)
        self._check_array_dims(y_train, 1, "y_train")
        self._check_array_dims(y_test, 1, "y_test")
        self._check_ytest_predproba_shapes(y_test, pred_proba)



    @staticmethod
    def _extract_info_from_data(
        pred_proba: list[np.ndarray],
        y_train: list[np.ndarray]
    ) -> list[list]:
        list_unique_tuples = [np.unique(array, return_counts=True) for array in y_train]
        classes = [t[0] for t in list_unique_tuples]
        classes_counts = [t[1] for t in list_unique_tuples]
        classification_setting = ["binary" if array.size == 2 else "multiclass" for array in classes]
        
        # allowing NA/nan/None in predictions
        pred_labels = [
            a 
            if not isinstance(a, np.ndarray) and pd.isna(a) 
            else np.argmax(a, axis=1) 
            for a in pred_proba
        ]

        return classes, classes_counts, classification_setting, pred_labels



    @staticmethod
    def _save_partial_data(partial_data: Any, save_path: str | Path) -> str:
        '''
        Saves the partial data object using the pickle module. 
        Returns the filepath as string.
        '''
        save_path = str(save_path)
        date_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
        dump_file = os.path.join(save_path, f"partial_data__{date_stamp}.pkl")
        with open(dump_file, "wb") as f:
            pickle.dump(partial_data, f)
        return dump_file



    @staticmethod
    def _compute_metrics(
        df: pd.DataFrame, 
        multiclass: Literal["ovr", "average"], 
        average_strategy: Literal["micro", "macro", "weighted"]
    ) -> pd.DataFrame:
        '''
        Internal version of 'compute_metrics' that act on a generic dataframe.
        Returns the extended dataframe.
        '''
        df_metrics = df.apply(
            compute_metrics, 
            axis=1, 
            multiclass=multiclass, 
            average_strategy=average_strategy
        )
        return pd.concat([df, df_metrics], axis=1)
    


    def compute_metrics(
        self, 
        multiclass: Literal["ovr", "average"],
        average_strategy: Literal["micro", "macro", "weighted"]
    ) -> "PredictionDataframe":
        '''
        It computes a series of metrics for each row of the prediction dataframe: 
        recall, precision, f1, accuracy, average_precision and auc.

        For the "multiclass" rows the metrics can be computed and returned
        in a "one vs rest" approach (ovr) or averaged using different strategies.
        
        In "ovr" multiclass scenario multiple values are computed and stored in numpy arrays.
        The values order of the resulting array follows the encoded numerical order of classes, 
        assuming the class at that position as the positive one. 
        So the first metric refer to the case in which the positive class is 0, then 1, 2, ... .

        In "average" multiclass scenario single averaged values are computed and returned.
        
        Parameters:
            multiclass (Literal["ovr", "average"]): 
                Whether to compute and remain statistics in one vs rest format,
                or apply an averaging strategy on them/data (them/data because
                some strategy like "micro" does not pass by this ovr metrics).
        
            average_strategy (Literal["micro", "macro", "weighted"]):
                Average strategy to use. Is ignored if multiclass is not "average"

        Returns: 
            Self.
        '''
        if self.df is None:
            raise ValueError("The 'df' attribute is None.") 
        self.df = self._compute_metrics(self.df, multiclass, average_strategy)
        return self
         


    def to_csv(self, filepath: str | Path, **kwargs) -> None:
        '''
        Save the underlying DataFrame into a text file.
        Parameters:
            filepath (str | Path): Filepath.
            **kwargs: 
                Additional keywords args to pass to "to_csv" pandas 
                method called on the underlying DataFrame.
        Returns: None
        '''
        if self.df is None:
            raise ValueError(
                "The PredictionDataframe does not contain data. The 'df' attribute is None."
            )
        
        df_copy = deepcopy(self.df)

        for col in self.all_columns_to_parse:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(save_ndarray_to_str)

        df_copy.to_csv(filepath, **kwargs)



    def build_from_file(
        self, 
        file: str | Path, 
        **read_params
    ) -> "PredictionDataframe":
        '''
        Read the prediction DataFrame from a file using pandas read_csv function.
        Parameters:
            file (str | Path): filepath.
            **read_params: Additonal kwargs to pass to the pandas "read_csv" function.
        Returns: 
            Self.
        '''
        df = pd.read_csv(file, **read_params)
        self._check_must_columns_presence(df)
        self._warn_na_in_must_columns(df)
        self.df = self._parse_str_to_numpy_arrays(df)
        return self



    def build_from_folder(
        self, 
        folder: str | Path, 
        glob_pattern: str = "*", 
        recursive: bool = False,   
        **read_params
    ) -> "PredictionDataframe":
        '''
        Build the prediction DataFrame from a folder using pandas read_csv function.
        
        Parameters:
            path (str | Path): 
                Path to the folder containing the text prediction dataframes files.
            glob_pattern (str, optional): 
                Pattern used to select the prediction dataframe files. Defaults to "*".
            recursive (bool, optional): 
                Whether to recursevely search in the folder specified in path.

        Returns: 
            self.
        '''
        folder = Path(folder) if isinstance(folder, str) else folder

        if not folder.exists():
            raise FileNotFoundError(f"'{folder}' folder is not found.")
        if not folder.is_dir():
            raise FileNotFoundError(f"'{folder}' is not a folder.")

        bound_search_method = folder.rglob if recursive else folder.glob
        dfs = []

        for df_file in bound_search_method(glob_pattern):
            dfs.append(pd.read_csv(df_file, **read_params))

        df = pd.concat(dfs, axis=0, ignore_index=True)
        self._check_must_columns_presence(df)
        self._warn_na_in_must_columns(df)
        df = self._parse_str_to_numpy_arrays(df)
        self.df = df
        return self



    def _parse_str_to_numpy_arrays(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Parse the string representation of numpy arrays back to numpy arrays'''
        for col in self.all_columns_to_parse:
            if col in df.columns:
                df[col] = df[col].apply(save_str_to_ndarray)
        return df

        
    def get_df(self) -> pd.DataFrame:
        '''Get the underlying DataFrame object.'''
        return self.df   


    @staticmethod
    def _check_ytest_predproba_shapes(
        y_test: Iterable[np.ndarray], 
        pred_proba: Iterable[np.ndarray]
    ) -> None:
        '''
        Checks whether y_test and pred_proba shapes are compatible.
        The check is done on the corresponding elements of the two iterables.
        '''
        for y, proba in zip(y_test, pred_proba):
            if not isinstance(proba, np.ndarray) and pd.isna(proba): 
                continue
            elif y.size != proba.shape[0]:
                raise ValueError(
                    "Found discrepancies in the shapes of 'y_test' and 'pred_proba' arrays."
                )


    @staticmethod
    def _check_array_dims(
        iterable: Iterable[np.ndarray], 
        number_dims: int, 
        name_iterable: str, 
        allow_na: bool = False
    ) -> None:
        '''Utility to check the number of dimensions of numpy arrays'''
        for array in iterable:
            if allow_na and not isinstance(array, np.ndarray) and pd.isna(array):
                continue
            elif len(array.shape) != number_dims:
                raise ValueError(f"Not all arrays in {name_iterable} are {number_dims}D.")


    def _check_must_columns_presence(self, df: pd.DataFrame) -> None:
        '''Perform sanity check on the prediction dataframe.'''
        for col in self.must_columns:
            if col not in df.columns:
                raise KeyError(
                    f"'{col}' column must be present in a PredictionDataFrame object."
                )
            

    def _warn_na_in_must_columns(self, df: pd.DataFrame) -> None:
        '''Warns about NAs in must columns'''
        for col in self.must_columns:
            if df[col].isna().any():
                warn(f"Found NAs in '{col}' columns.")