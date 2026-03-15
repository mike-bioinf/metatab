import pandas as pd
from copy import deepcopy
from pathlib import Path
from metatab._paper.analysis.utils import check_presence_cols



def correct_tabpfn_metadata(metadata: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    '''
    Apply corrections to the tabpfn metadata. In detail:
    1) Remove the root path of the "model_path" column values.
    2) Convert the list string representation of "inference_config__PREPROCESS_TRANSFORMS"
    to a tuple string representation.

    Parameters:
        metadata (pd.DataFrame): tabpfn metadata to correct.
        copy (bool, optional): Whether to create and modify a deepcopy of metadata

    Returns:
        pd.DataFrame: Returns the corrected metadata.
    '''
    check_presence_cols(metadata, cols=["model_path", "inference_config__PREPROCESS_TRANSFORMS"])
    
    metadata = deepcopy(metadata) if copy else metadata
    metadata["model_path"] = metadata["model_path"].apply(lambda x: str(Path(x).name))
    
    def change_parentesis(x: str) -> str:
        x = x[1:-1]
        # we have to add the comma if the original tuple is of length one
        if x.count("{") > 1:
            return "(" + x + ")"
        else:
            return "(" + x + ",)"
    
    metadata["inference_config__PREPROCESS_TRANSFORMS"] = metadata["inference_config__PREPROCESS_TRANSFORMS"].apply(
        change_parentesis        
    )

    return metadata