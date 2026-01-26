import pandas as pd
from importlib.resources import files


def get_example_data() -> pd.DataFrame:
    '''
    Get a microbial taxonomic profile sourced from the `curatedMetagenomicData` repository. 
    It corresponds to an older MetaPhlAn-based profiling (MetaPhlAn v3).

    The target variable is stored in the column `"Group"`, which contains two classes:
    - `"control"` (n=14)
    - `"schizophrenia"` (n=15)
    
    Returns:
        pd.DataFrame: The example dataset.
    '''
    data_path = files("metatab").joinpath("data/castronaller_2015.txt")
    return pd.read_csv(data_path, sep="\t")



def get_example_data_path() -> str:
    '''
    Get the example microbial profile absolute path.
    '''
    return str(files("metatab").joinpath("data/castronaller_2015.txt"))