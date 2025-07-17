import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from typing import Literal
from pathlib import Path
from copy import deepcopy
from warnings import warn
from utils.prediction.parser import parse_pred_dataframe
from utils.prediction.metrics import compute_metrics

from utils.prediction.constants import (
    PERFORMANCE_METRICS,
    PARSE_COLUMNS,
    MANDATORY_COLUMNS
)

from utils.prediction.utils import (
    wrap_into_list, 
    to_numpy_iterable, 
    are_same_length
)



class PredictionDataframe():
    '''
    Class to manage the I/O of classification prediction data organized in a pandas DataFrame.

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
        self.columns_to_parse = PARSE_COLUMNS
        self.metrics_columns = PERFORMANCE_METRICS
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
        if an exception is raised during execution, 
        in order to not lose previous computation (see 'save_path' param).

        Parameters:
            dataset (str | list[str] | pd.Series): Dataset name/s.
            y_train (np.ndarray | pd.Series | list[pd.Series | np.ndarray]): Training labels.
            y_test (np.ndarray | pd.Series | list[pd.Series | np.ndarray]): Test labels.
            pred_proba (np.ndarray | list[np.ndarray]): Prediction probabilities.
            save_path (str | Path | None): 
                Path to the folder where the pickle file is saved.
                The file name is automatically created using the "partial_data__" prefix + time stamp.
                If None, the default, the recover strategy is not used.
            **add_columns: Additional columns to add to the DataFrame.

        Returns:
            Self, fhe dataframe is stored in self.df.
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
                
            dataset, y_train, y_test, pred_proba = wrap_into_list(
                dataset, 
                y_train, 
                y_test, 
                pred_proba
            )
            
            y_train = to_numpy_iterable(y_train)
            y_test = to_numpy_iterable(y_test)
            
            if not are_same_length(dataset, y_train, y_test, pred_proba):
                raise ValueError((
                    "The input iterables have not the same length." 
                    " Note that 'scalar' inputs are internally converted to iterables."
                ))
            
            self._check_array_dims(pred_proba, 2, "pred_proba", allow_na=True)
            self._check_array_dims(y_train, 1, "y_train")
            self._check_array_dims(y_test, 1, "y_test")
            self._check_ytest_predproba_shapes(y_test, pred_proba)
            
            list_unique_tuples = [np.unique(array, return_counts=True) for array in y_train]
            classes = [t[0] for t in list_unique_tuples]
            classes_counts = [t[1] for t in list_unique_tuples]
            classification_setting = ["binary" if array.size == 2 else "multiclass" for array in classes]
            
            # allowing NAs/nan/None in predictions
            pred_labels = [
                a 
                if not isinstance(a, np.ndarray) and pd.isna(a) 
                else np.argmax(a, axis=1) 
                for a in pred_proba
            ]

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
                print(f"Caught Exception in the PredictionDataframe building process. Recovering.", file=sys.stderr)
                print(f"Partial data saved to: {dump_file}", file=sys.stderr)
            raise
            


    @staticmethod
    def _save_partial_data(partial_data, save_path) -> str:
        '''Saves the partial data object using the pickle module. Returns the filepath as string.'''
        save_path = str(save_path)
        date_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
        dump_file = os.path.join(save_path, f"partial_data__{date_stamp}.pkl")
        with open(dump_file, "wb") as f:
            pickle.dump(partial_data, f)
        return dump_file



    def build_from_file(
        self, 
        file: str | Path, 
        parse: bool = False, 
        **read_params
    ) -> "PredictionDataframe":
        '''
        Read the prediction DataFrame from a file using pandas read_csv function.
        Parameters:
            file (str | Path): filepath.
            **read_params: 
                Additonal kw args to pass to the pandas "read_csv" function.
            parse (bool, optional): 
                Whether to parse the "classes", "classes_counts", "test_labels", 
                "pred_labels" and "pred_proba" columns from string to numpy arrays.
                
        Returns: 
            Self.
        '''
        df = pd.read_csv(file, **read_params)
        self._check_must_columns_presence(df)
        self._warn_na_in_must_columns(df)
        if parse: df = parse_pred_dataframe(df)
        self.df = df
        return self



    def build_from_folder(
        self, 
        folder: str | Path, 
        glob_pattern: str = "*", 
        recursive: bool = False, 
        parse: bool = False,  
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
            parse (bool, optional): 
                Whether to parse the "classes", "classes_counts", "test_labels", 
                "pred_labels" and "pred_proba" columns from string to numpy arrays.

        Returns: 
            self.
        '''
        folder = Path(folder) if isinstance(folder, str) else folder

        if not folder.exists():
            raise FileNotFoundError(f"'{folder}' folder is not found.")
        if not folder.is_dir():
            raise FileNotFoundError(f"'{folder}' is not a folder.")

        search_bound_method = folder.rglob if recursive else folder.glob

        dfs = []
        for df_file in search_bound_method(glob_pattern):
            dfs.append(pd.read_csv(df_file, **read_params))

        df = pd.concat(dfs, axis=0, ignore_index=True)
        self._check_must_columns_presence(df)
        self._warn_na_in_must_columns(df)
        if parse: df = parse_pred_dataframe(df)
        self.df = df
        return self



    def get_df(self) -> pd.DataFrame:
        '''Get the underlying DataFrame object.'''
        return self.df   



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
        
        df_metrics = self.df.apply(
            compute_metrics, 
            axis=1, 
            multiclass=multiclass, 
            average_strategy=average_strategy
        )
        
        self.df = pd.concat([self.df, df_metrics], axis=1)
        return self
         


    def to_csv(self, filepath: str | Path, precision: int = 16, **kwargs) -> None:
        '''
        Save the underlying DataFrame into a text file.
        Parameters:
            filepath (str | Path): Filepath.
            precision (int, optional): 
                Number of digits used to store numpy arrays values.
            **kwargs: 
                Additional keywords args to pass to "to_csv" pandas 
                method called on the underlying DataFrame
        Returns: None
        '''
        if self.df is None:
            raise ValueError(
                "The PredictionDataFrame instance does not contain data. self.df is None."
            )
        
        df_copy = deepcopy(self.df)

        for col in (self.columns_to_parse + self.metrics_columns):
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(self._save_array2string, precision=precision)

        df_copy.to_csv(filepath, **kwargs)


    @staticmethod
    def _save_array2string(value, precision: int):
        '''array2string numpy function with conditional checking on input type.'''
        return value \
            if not isinstance(value, np.ndarray) \
            else np.array2string(value, precision=precision, floatmode="fixed", threshold=sys.maxsize)


    @staticmethod
    def _check_ytest_predproba_shapes(y_test: list[np.ndarray], pred_proba: list[np.ndarray]) -> None:
        '''Checks whether y_test and pred_proba shapes are compatible'''
        for y, proba in zip(y_test, pred_proba):
            if not isinstance(proba, np.ndarray) and pd.isna(proba): 
                continue
            elif y.size != proba.shape[0]:
                raise ValueError("Found discrepancies in the shapes of 'y_test' and 'pred_proba' arrays.")


    @staticmethod
    def _check_array_dims(
        iterable: list[np.ndarray], 
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
