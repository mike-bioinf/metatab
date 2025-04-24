import re
import pandas as pd
from pathlib import Path



def load_data_df_mode(path: str | Path, target_feature: str) -> tuple[pd.DataFrame | pd.Series]:
    '''Loads data in 'df' input mode'''
    path = path if isinstance(path, Path) else Path(path)
    df = pd.read_csv(path, sep="\t")
    X = df.drop(columns=target_feature)
    y = df[target_feature]
    return X, y



def load_data_xy_mode(path: str | Path, **kwargs) -> tuple[pd.DataFrame | pd.Series]:
    '''Loads data in 'xy' input mode'''
    path = path if isinstance(path, Path) else Path(path)
    X = pd.read_csv(path / "X.txt", sep="\t")
    y = pd.Series(pd.read_csv(path / "y.txt", sep="\t").iloc[:, 0])
    return X, y



def load_data_sets_mode(
        path: str | Path, 
        save_missing: bool = False
    ) -> dict[str, pd.DataFrame | pd.Series]:
    '''
    Utility to load the X and y training and testing sets for a dataset.
    The function assumes that the sets files are named following the convention : "X/y_train/test.txt".
    These files must be located at the path specificied in 'path'.

    Parameters:
        path (str | Path): Path where the sets files live.
        save_missing (bool): 
            If True one can avoid errors for missing sets. When False the function will look
            for all x and y train and test sets and will raise an error if some are missing.
            Defaults to False.
    
    Returns:
        dict[str,pd.DataFrame|pd.Series]: 
        A dictionary where the keys specify the type of set, i.e. one of "X/y_train/val/test",
        and the values are the actual sets as pandas Dataframe or Series for X and y respectively.
    '''
    path = Path(path) if isinstance(path, str) else path
    sets_names = ["X_train", "X_test", "y_train", "y_test"]
    dict_sets = {}
    
    for set_name in sets_names:
        set_path = path / f"{set_name}.txt"
        if not set_path.exists():
            if save_missing:
                continue
            else:
                raise FileNotFoundError(f"The file '{set_path}' does not exist.")
        set_value = pd.read_table(set_path, sep="\t")
        if re.match("y_", set_name):
            set_value = pd.Series(set_value.iloc[:, 0])
        dict_sets[set_name] = set_value

    return dict_sets