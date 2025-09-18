from typing import Literal
from copy import deepcopy
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from hp_search.tabpfn_search_space import (
    enumerate_preprocess_transforms, 
    return_clf_paths_list
)




COLUMN_TRANSFORMER_FIXED_PARAMS = {
    "remainder": "passthrough",
    "n_jobs": 1, # avoid cluster cores problem
    "force_int_remainder_cols": False, # to suppress a FutureWarning
    "sparse_threshold": 0  # to avoid output conversion to sparse matrix objects 
}


HPS_ENCODING_SCHEME_RANDOM_FOREST = ColumnTransformer(
    transformers=[
        (
            "onehot",
            # we cast and save this col as str
            OneHotEncoder(categories=[["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "None", "sqrt", "log2"]]), 
            ["max_features"]
        )
    ],
    ** COLUMN_TRANSFORMER_FIXED_PARAMS
)


HPS_ENCODING_SCHEME_TABPFN = ColumnTransformer(
    transformers=[
        (  
            "onehot",
            OneHotEncoder(
                categories=[
                    # we save the lists as string in the meta-data so it should work 
                    [str(list_of_dicts) for list_of_dicts in enumerate_preprocess_transforms()],
                    # The float are saved as str(float) that should return the representation below
                    ["None", "7.0", "9.0", "12.0"],
                    return_clf_paths_list()
                ]
            ),
            [
                "inference_config__PREPROCESS_TRANSFORMS", 
                "inference_config__OUTLIER_REMOVAL_STD", 
                "model_path"
            ]
        ),
        (
            "binary",
            OrdinalEncoder(categories=[["0.99", "None"]]),
            ["inference_config__SUBSAMPLE_SAMPLES"]
        )
    ],
    **COLUMN_TRANSFORMER_FIXED_PARAMS
)


## TODO: complete once defined default tuninng space
HPS_ENCODING_SCHEME_ESTIMATORS = {
    "random_forest": HPS_ENCODING_SCHEME_RANDOM_FOREST,
    "xgb": None, ## complete
    "catboost": None,  ## complete
    "lgbm": None,
    "tabpfn": HPS_ENCODING_SCHEME_TABPFN
}


def get_encoding_scheme(estimator: Literal["random_forest", "xgb", "catboost", "lgbm", "tabpfn"]) -> None | ColumnTransformer:
    '''
    Get a deepcopy of the encoding scheme of the HP feature space 
    designed for the input estimator.
    '''
    return deepcopy(HPS_ENCODING_SCHEME_ESTIMATORS[estimator])