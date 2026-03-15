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

    The class implements also generic/train/test_name attributes which 
    refer to the loaded dataset names. These are euristically determined.
    '''
    def __init__(self):
        self.X: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y: pd.Series = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.generic_dataset_name: str = None
        self.train_dataset_name: str = None
        self.test_dataset_name: str = None

    
    def load(self, mode: Literal["df", "xy", "sets"], **load_params) -> None:
        '''
        Allow to call a specific 'load' method using the mode parameter.
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
    ) -> None:
        '''
        Loads data from an input file ('df' input mode).
        Set the name of the dataset as the filename without extension.

        Parameters:
            path (str | Path): Path of the file to load.
            target_feature (str): Name of the y column.
            load_as (Literal["generic", "train", "test"]): 
                Specify how to store the data.
            kwargs: 
                Ignored. Ensures compatibility with other load methods.
        '''
        path = path if isinstance(path, Path) else Path(path)
        dname = path.stem
        
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exists.")
        
        df = pd.read_csv(path, sep="\t")

        if not target_feature in df.columns:
            raise KeyError(f"'{target_feature}' column is not found in the dataframe.")
        
        X = df.drop(columns=target_feature)
        y = df[target_feature]
        self._set_xyd_attributes(X, y, dname, load_as)


    def load_xy_mode(
        self, 
        path: str | Path, 
        load_as: Literal["generic", "train", "test"], 
        **kwargs
    ) -> None:
        '''
        Loads data in 'xy' input mode. 
        The function assumes that the X and y files are named 
        "X.txt" and "y.txt" respectively.
        
        These files must be located at the path specificied in 'path',
        which last folder is choosen as dataset name.
        
        Parameters:
            path (str, Path): 
                Path to the directory containing the data files.
            load_as (Literal["generic", "train", "test"]): 
                Specify how to store the data.
            kwargs: 
                Ignored. Ensures compatibility with other load methods.
        '''
        path = path if isinstance(path, Path) else Path(path)
        dname = path.stem
        x_path = path / "X.txt"
        y_path = path / "y.txt"

        for p in [path, x_path, y_path]:
            if not p.exists:
                raise FileNotFoundError(f"The path '{p}' does not exists.")

        X = pd.read_csv(x_path, sep="\t")
        y = pd.Series(pd.read_csv(y_path, sep="\t").iloc[:, 0])
        self._set_xyd_attributes(X, y, dname, load_as)


    def load_sets_mode(
        self,
        path: str | Path,
        skip: str | list[str] | None = None,
        save_missing: bool | str | list[str] = False,
        **kwargs
    ) -> None:
        '''
        Loads X/y train/test sets (so no load_as capability). 
        
        Assumes that the sets files are named following the convention:
        "X/y_train/test.txt".
        
        These files must be located at the path specificied in 'path',
        which last folder is stored as 'generic_dataset_name'.

        Parameters:
            path (str | Path): 
                Path to the directory containing the data files. 
            
            skip (str | list[str]) | None:
                String or list of strings that specifies the sets that must not be loaded.
                If None the function tries to load all sets.

            save_missing (bool | str | list[str], optional): 
                - If False (default), raise an error if a required file is missing.
                - If True, silently skip any missing file.
                - If a str or list of str (from "X_train", "X_test", "y_train", "y_test"), 
                does not raise errors for the missingness of the specified sets,
                which can be or cannot be loaded depending on their existence.
            
            kwargs: 
                Ignored. Ensures compatibility with other load methods.
        '''
        path = Path(path) if isinstance(path, str) else path
        dname = path.stem
        setattr(self, "generic_dataset_name", dname)

        save_missing = [save_missing] if isinstance(save_missing, str) else save_missing
        skip = [] if skip is None else skip
        skip = skip if isinstance(skip, list) else [skip]
        sets_names = ["X_train", "X_test", "y_train", "y_test"]
        
        for set_name in sets_names:
            if set_name in skip:
                continue
            
            set_path = path / f"{set_name}.txt"
            
            if not set_path.exists():
                if (
                    (isinstance(save_missing, bool) and save_missing) or 
                    (isinstance(save_missing, list) and set_name in save_missing)
                ):
                    continue
                else:
                    raise FileNotFoundError(f"The file '{set_path}' does not exist.")
            
            data = pd.read_table(set_path, sep="\t")

            if set_name.startswith("y_"):
                data = pd.Series(data.iloc[:, 0])
            
            setattr(self, set_name, data)


    def _set_xyd_attributes(
        self, 
        x_value: Any, 
        y_value: Any,
        dname_value: str,
        load_type: Literal["generic", "train", "test"]
    ) -> None:
        '''
        Set the input x, y, dataset_name values in the 
        correct attributes based on load_type.
        '''
        x_attr, y_attr, name_attr = self._get_names_xyd_attributes(load_type)        
        setattr(self, x_attr, x_value)
        setattr(self, y_attr, y_value)
        setattr(self, name_attr, dname_value)


    @staticmethod
    def _get_names_xyd_attributes(load_type: Literal["generic", "train", "test"]) -> tuple[str, str, str]:
        if load_type == "generic":
            return "X", "y", "generic_dataset_name"
        elif load_type == "train":
            return "X_train", "y_train", "train_dataset_name"
        else:
            return "X_test", "y_test", "test_dataset_name"