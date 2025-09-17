from typing import Literal
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from hp_search.tabpfn_search_space import (
    enumerate_preprocess_transforms, 
    return_clf_paths_list
)




HPS_ENCODING_SCHEME_RANDOM_FOREST = ColumnTransformer(
    transformers=[
        (
            "max_features",
            # as string since we cast this column to str
            OneHotEncoder(categories=[["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "None", "sqrt", "log2"]]), 
            "max_features"
        )
    ],
    remainder="passthrough",
    n_jobs=1 # avoid cluster cores problem
)



HPS_ENCODING_SCHEME_TABPFN = ColumnTransformer(
    transformers=[
        (  
            "onehot",
            OneHotEncoder(
                categories=[
                    # we save the lists as string in the meta-data so it should work 
                    [str(list_of_dicts) for list_of_dicts in enumerate_preprocess_transforms()],
                    # The float are saved as str(float) that should return the below categories
                    ["None", "7.0", "9.0", "12.0"],
                    return_clf_paths_list()
                ]
            ),
            [
                "inference_config__PREPROCESS_TRANSFORMS", 
                "inference_config__OUTLIER_REMOVAL_STD", 
                "model_path"
            ]
        )
    ],
    remainder="passthrough",
    n_jobs=1 # avoid cluster cores problem
)



## TODO: complete once defined default tuninng space
HPS_ENCODING_SCHEME_ESTIMATORS = {
    "random_forest": HPS_ENCODING_SCHEME_RANDOM_FOREST,
    "xgb": None,
    "catboost": None,
    "lgbm": None,
    "tabpfn": HPS_ENCODING_SCHEME_TABPFN
}



def get_encoding_scheme(estimator: Literal["random_forest", "xgb", "catboost", "lgbm"]) -> ColumnTransformer:
    '''
    Get a deepcopy of the ColumnTransformer template instance 
    designed for the HPs space of the input estimator.
    '''
    return deepcopy(HPS_ENCODING_SCHEME_ESTIMATORS[estimator])