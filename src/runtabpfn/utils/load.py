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