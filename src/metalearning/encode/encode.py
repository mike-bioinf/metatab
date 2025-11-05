"""
In this module we set the encoding (preprocessing) to apply to the meta-data obtained 
from the different estimators, considering ONLY the default estimators tune spaces.

Some general indications:
- For the gbdts the "es" version uses the same tune space of the base ones.
This means that we define just one encoding for both.
- We must transform some numerical or mixed typed columns to str since we want to encode 
them though the sklearn ordinal or onehot encoders, which require input categories of the
same type.
- We expect some metafeatures to contain full nan values. This is because pymfe does not
automatically filter the structurally incompatible metafeature-summary_func combinations.
These are addressed in practice by the VarianceThreshold step which is able to remove 
full nan features.
- We expect some metafeatures to contain data-dependent nan, i.e. nan values that
compares only on some cases. We do not take care of this nan values since
the RandomForestRegressor (our surrogate model) is able to natively handle them.
- Some hyperparameters have nan values that should be converted back to None. 
This is due to pandas IO behaviour. Either the case we resolve this by implementing 
the NanToNone transformer which is able to work on specific columns only.
In this way we avoid to erroneously apply this conversion on the metafeatures.
"""

from copy import deepcopy
from typing import Literal
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from metalearning.encode.transformers import NanToNone, ColToStr

from hp_search.tabpfn_search_space import (
    enumerate_preprocess_transforms,
    TABPFN_CHECKPOINTS
)




COLUMN_TRANSFORMER_FIXED_PARAMS = {
    "remainder": "passthrough",
    # avoid cluster cores/threads problem
    "n_jobs": 1,
    # use False to suppress a FutureWarning (the parameters will be deprecated in sklearn v 1.9)
    "force_int_remainder_cols": False,
    # avoid output conversion to sparse matrix objects
    "sparse_threshold": 0,
    # avoid the addition of the name of the transformed used to generate the new column
    "verbose_feature_names_out": False
}


PREPROCESSING_COLUMN_ENCODING = (
    "preprocessing_column", 
    OneHotEncoder(categories=[["base", "pca", "density_filter"]], sparse_output=False),
    ["preprocessing"]
)


def create_preprocessing_encoding() -> ColumnTransformer:
    '''Create a ColumnTransformer executing the "preprocessing" column encoding only.'''
    return ColumnTransformer(
        transformers=[PREPROCESSING_COLUMN_ENCODING],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    )




HPS_ENCODING_SCHEME_RANDOM_FOREST = [
    NanToNone("max_features", check_on_fit=True), 
    ColToStr("max_features", check_on_fit=True),
    ColumnTransformer(
        transformers=[
            (
                "onehot",
                # we cast and save this col as str
                OneHotEncoder(
                    categories=[["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "None", "sqrt", "log2"]],
                    sparse_output=False
                ), 
                ["max_features"]
            ),
            PREPROCESSING_COLUMN_ENCODING
        ],
        ** COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_TABPFN = [
    NanToNone(
        columns=[
            "inference_config__OUTLIER_REMOVAL_STD", 
            "inference_config__SUBSAMPLE_SAMPLES"
        ], 
        check_on_fit=True
        ),
    ColToStr(
        columns=[
            "inference_config__OUTLIER_REMOVAL_STD", 
            "inference_config__SUBSAMPLE_SAMPLES", 
            "inference_config__PREPROCESS_TRANSFORMS"
        ],
        check_on_fit=True
    ),
    ColumnTransformer(
        transformers=[
            (  
                "onehot",
                OneHotEncoder(
                    categories=[
                        # we have the lists as tuple string representation in the metadata 
                        [str(tuple(list_of_dicts)) for list_of_dicts in enumerate_preprocess_transforms()],
                        ["None", "7.0", "9.0", "12.0"],
                        TABPFN_CHECKPOINTS
                    ],
                    sparse_output=False
                ),
                [
                    "inference_config__PREPROCESS_TRANSFORMS", 
                    "inference_config__OUTLIER_REMOVAL_STD", 
                    "model_path"
                ]
            ),
            (
                "binary",
                OrdinalEncoder(categories=[["0.99", "None"], ["no"]]),
                ["inference_config__SUBSAMPLE_SAMPLES", "inference_config__POLYNOMIAL_FEATURES"]
            ),
            PREPROCESSING_COLUMN_ENCODING
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_XGB = [
    ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        ["depthwise"],
                        ["exact"]
                    ]
                ),
                [
                    "grow_policy",
                    "tree_method"
                ]
            ),
            PREPROCESSING_COLUMN_ENCODING
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_CATBOOST = [
    ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        ["Cosine"],
                        ["SymmetricTree"],
                        ["Plain"]
                    ]
                ),
                [
                    "score_function",
                    "grow_policy",
                    "boosting_type"
                ]
            ),
            PREPROCESSING_COLUMN_ENCODING
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_LGBM = [
    create_preprocessing_encoding(),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME = {
    "random_forest": HPS_ENCODING_SCHEME_RANDOM_FOREST,
    "xgb": HPS_ENCODING_SCHEME_XGB,
    "catboost": HPS_ENCODING_SCHEME_CATBOOST,
    "lgbm": HPS_ENCODING_SCHEME_LGBM,
    "tabpfn": HPS_ENCODING_SCHEME_TABPFN
}


def get_encoding_scheme(
    estimator: Literal["random_forest", "xgb", "catboost", "lgbm", "tabpfn"]
) -> list:
    '''
    Get a deepcopy of the encoding scheme of the HP feature space designed for the input estimator.
    The encoding scheme consists in a ordered list of sklearn transformers to insert in a Pipeline object.
    '''
    return deepcopy(HPS_ENCODING_SCHEME[estimator])