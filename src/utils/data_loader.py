import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Any



class DataLoader():
    '''
    Abstract the data loading logic.

    Supports multiple input formats:
    - sets: A folder containing files named according to the convention `X/y_train/test.txt`.
    - xy: A folder containing `X.txt` and `y.txt` files.
    - df: A single text file containing both X and y data (in this case one must specify the target feature).
    
    Allows the user to specify the "nature" of the data to load.

    The class implements {X/y}{ /_train/_test) attributes:
    - X/y represent the data from which the train/test sets are still to be derived.
    - The X/y train/test sets are the ones directly to use in training and testing.
    '''
    
    def __init__(self):
        self.X: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y: pd.Series = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None

    

    def return_train_test_xy_sets(
        self, 
        train_idx: np.ndarray = None, 
        test_idx: np.ndarray = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        '''
        Returns the instance train test sets if available (not None),
        otherwise derive them from X, y using the input indexes.
        Note: in case the train and test sets are available the input indexes
        are ignored and therefore they can be None or any other value.
        '''
        X_train = self.X_train if self.X_train is not None else self.X.iloc[train_idx, :]
        y_train = self.y_train if self.y_train is not None else self.y.iloc[train_idx]
        X_test = self.X_test if self.X_test is not None else self.X.iloc[test_idx, :]
        y_test = self.y_test if self.y_test is not None else self.y.iloc[test_idx]
        return X_train, y_train, X_test, y_test



    def load(self, mode: Literal["df", "xy", "sets"], **load_params) -> None:
        '''
        Allow to call a specific 'load' method using the mode attribute.
        One must pass all the specific load method parameters.
        '''
        if mode == "df":
            self.load_df_mode(**load_params)
        elif mode == "xy":
            self.load_xy_mode(**load_params)
        elif mode == "sets":
            self.load_sets_mode(**load_params)
        else:
            raise ValueError(
                f"mode must be one of 'df', 'xy' and 'sets'. '{mode}' is not supported."
            )


    
    def load_df_mode(
        self,
        path: str | Path, 
        target_feature: str, 
        load_as: Literal["generic", "train", "test"], 
        **kwargs
    ):
        '''
        Loads data from an input file ('df' input mode).

        Parameters:
            path (str | Path): Path of the file to load.
            target_feature (str): Name of the y column.
            load_as (Literal["generic", "train", "test"]): 
                Specify how to store the data.
            kwargs: 
                Does nothing except ensuring compability with other load functions.
        '''
        path = path if isinstance(path, Path) else Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exists.")
        
        df = pd.read_csv(path, sep="\t")

        if not target_feature in df.columns:
            raise KeyError(f"'{target_feature}' column is not found in the dataframe.")
        
        X = df.drop(columns=target_feature)
        y = df[target_feature]
        self._set_xy_attributes(X, y, load_as)



    def load_xy_mode(
        self, 
        path: str | Path, 
        load_as: Literal["generic", "train", "test"], 
        **kwargs
    ) -> None:
        '''
        Loads data in 'xy' input mode.The function assumes that the X and y files 
        are named "X.txt" and "y.txt" respectively.
        These files must be located at the path specificied in 'path'.
        
        Parameters:
            path (str, Path): Path where the x and y files live.
            load_as (Literal["generic", "train", "test"]): 
                Specify how to store the data.
            kwargs: 
                Does nothing except ensuring compability with other load functions.
        '''
        path = path if isinstance(path, Path) else Path(path)
        x_path = path / "X.txt"
        y_path = path / "y.txt"

        for p in [path, x_path, y_path]:
            if not p.exists:
                raise FileNotFoundError(f"The path '{p}' does not exists.")

        X = pd.read_csv(x_path, sep="\t")
        y = pd.Series(pd.read_csv(y_path, sep="\t").iloc[:, 0])
        self._set_xy_attributes(X, y, load_as)



    def load_sets_mode(
        self,
        path: str | Path, 
        save_missing: bool | str | list[str] = False,
        **kwargs
    ) -> None:
        '''
        Loads X/y train/test sets (so no load_as capability). 
        Assumes that the sets files are named following the convention : "X/y_train/test.txt".
        These files must be located at the path specificied in 'path'.

        Parameters:
            path (str | Path): Path where the sets files live.
            save_missing (bool | list[str], optional): 
                Either a boolean or a string or list of strings.
                - If boolean indicates whether the function should not raise errors for missing sets.
                If False, the default, the function will look for all x and y train and test sets and will 
                raise an error if some are missing.
                - If string or list of strings it saves the specified missing sets. This list should 
                therefore contains the values: "X_train", "X_test", "y_train" and "y_test".
            kwargs: Does nothing except ensuring compability with other load functions.
        '''
        path = Path(path) if isinstance(path, str) else path
        sets_names = ["X_train", "X_test", "y_train", "y_test"]
        save_missing = [save_missing] if isinstance(save_missing, str) else save_missing
        
        for set_name in sets_names:
            set_path = path / f"{set_name}.txt"
            
            if not set_path.exists():
                if (
                    (isinstance(save_missing, bool) and save_missing) or 
                    (isinstance(save_missing, list) and set_name in save_missing)
                ):
                    continue
                else:
                    raise FileNotFoundError(f"The file '{set_path}' does not exist.")
            
            set_value = pd.read_table(set_path, sep="\t")
            if re.match("y_", set_name):
                set_value = pd.Series(set_value.iloc[:, 0])

            setattr(self, set_name, set_value)



    def _set_xy_attributes(
        self, 
        x_value: Any, 
        y_value: Any, 
        load_type: Literal["generic", "train", "test"]
    ) -> None:
        '''Set the input x, y values in the correct X/y attributes based on load_type'''
        x_attr, y_attr = self._get_names_xy_attributes(load_type)        
        setattr(self, x_attr, x_value)
        setattr(self, y_attr, y_value)



    @staticmethod
    def _get_names_xy_attributes(load_type: Literal["generic", "train", "test"]) -> tuple[str, str]:
        if load_type == "generic":
            return "X", "y"
        elif load_type == "train":
            return "X_train", "y_train"
        else:
            return "X_test", "y_test"