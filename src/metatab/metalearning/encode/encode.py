"""
In this module we set the encoding (preprocessing) to apply to the meta-data obtained 
from the different estimators, considering ONLY the default estimators tune spaces.

Some general indications:
- For the gbdts the "es" versions uses the same tune space of the base ones.
This means that we can define just one encoding for both.

- We must transform some numerical or mixed typed columns to str since we want to encode 
them though the sklearn ordinal or onehot encoders, which require input categories of homogenous type.

- We expect some metafeatures to goes to +-inf due to our dataset statistical properties.
We deal with this by employing the InfToNan transformer, which transform the +/-inf values
to nan. This is a good solution since our surrogate model (RandomForestRegressor) si able
to natively learn and handle nan value, both at training and inference time.

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
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from metatab.metalearning.encode.transformers import NanToNone, ColToStr, InfToNan
from metatab.classifiers.tabpfn import enumerate_preprocess_transforms, TABPFN_CHECKPOINTS
from metatab.utils.types import TunableEstimatorType



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


PREPROCESSING_COLUMN_TRANSFORMER = (
    "preprocessing_column", 
    OneHotEncoder(
        categories=[["no", "base", "pca", "density_filter"]], 
        handle_unknown="ignore", # to maintain model functionality when we add more preprocessig options
        sparse_output=False
    ),
    ["preprocessing"]
)


def create_preprocessing_encoding() -> ColumnTransformer:
    '''Create a ColumnTransformer executing the "preprocessing" column encoding only.'''
    return ColumnTransformer(
        transformers=[PREPROCESSING_COLUMN_TRANSFORMER],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    )



HPS_ENCODING_SCHEME_RANDOM_FOREST = [
    NanToNone("max_features"), 
    ColToStr("max_features"),
    InfToNan(),
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
            (
                "ordinal",
                OrdinalEncoder(categories=[["gini", "entropy"]]),
                ["criterion"]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        ** COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_EXTRA_TREES = [
    NanToNone("max_features"), 
    ColToStr("max_features"),
    InfToNan(),
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
            (
                "binary",
                OrdinalEncoder(categories=[["gini", "entropy"]]),
                ["criterion"]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        ** COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_TABPFN = [
    NanToNone([
        "inference_config__OUTLIER_REMOVAL_STD", 
        "inference_config__SUBSAMPLE_SAMPLES"
    ]),
    ColToStr([
        "inference_config__OUTLIER_REMOVAL_STD", 
        "inference_config__SUBSAMPLE_SAMPLES", 
        "inference_config__PREPROCESS_TRANSFORMS"
    ]),
    InfToNan(),
    ColumnTransformer(
        transformers=[
            (  
                "onehot",
                OneHotEncoder(
                    categories=[
                        # we have the lists string representation in metadata
                        [str(list_of_dicts) for list_of_dicts in enumerate_preprocess_transforms()],
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
                OrdinalEncoder(categories=[["0.99", "None"]]),
                ["inference_config__SUBSAMPLE_SAMPLES"]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_XGB = [
    InfToNan(),
    ColumnTransformer(
        transformers=[
            (
                "binary",
                OrdinalEncoder(categories=[["depthwise", "lossguide"]]),
                ["grow_policy"]
            ),
            (
                "onehot",
                OneHotEncoder(
                    categories=[["exact", "hist", "approx"]], 
                    sparse_output=False
                ),
                ["tree_method"]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_CATBOOST = [
    InfToNan(),
    ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(
                    categories=[["SymmetricTree", "Depthwise", "Lossguide"]],
                    sparse_output=False,
                ),
                ["grow_policy"]
            ),
            (
                "binary",
                OrdinalEncoder(
                    categories=[
                        ["Cosine", "L2"],
                        ["Plain", "Ordered"]
                    ]
                ),
                [
                    "score_function",
                    "boosting_type"
                ]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_LGBM = [
    InfToNan(),
    create_preprocessing_encoding(),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_REALMLP = [
    InfToNan(),
    ColToStr(["batch_size", "tfms"]),
    ColumnTransformer(
        transformers=[
            (
                "ordinal", 
                OrdinalEncoder(
                    categories=[
                        ["256", "auto"],
                        # tuple string represenation since the metadata is not hyperopt corrected
                        ["()", "('median_center', 'robust_scale', 'smooth_clip')"]
                    ]
                ),
                ["batch_size", "tfms"]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]


HPS_ENCODING_SCHEME_TABM = [
    InfToNan(),
    ColToStr(["batch_size", "tfms"]),
    ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        ["tabm", "tabm-mini"],
                        ["256", "auto"]
                    ]
                ),
                ["arch_type", "batch_size"]
            ),
            (
                "onehot",
                OneHotEncoder(
                    categories=[[str(l) for l in [[], ["quantile_tabr"], ["median_center", "robust_scale", "smooth_clip"]]]],
                    sparse_output=False
                ),
                ["tfms"]
            ),
            PREPROCESSING_COLUMN_TRANSFORMER
        ],
        **COLUMN_TRANSFORMER_FIXED_PARAMS
    ),
    VarianceThreshold()
]



# The "es" estimator version uses the same encoding 
# of their "base" counterpart, since they share the tune spaces
HPS_ENCODING_SCHEME = {
    "random_forest": HPS_ENCODING_SCHEME_RANDOM_FOREST,
    "extra_trees": HPS_ENCODING_SCHEME_EXTRA_TREES,
    "xgb": HPS_ENCODING_SCHEME_XGB,
    "es_xgb": HPS_ENCODING_SCHEME_XGB,
    "catboost": HPS_ENCODING_SCHEME_CATBOOST,
    "es_catboost": HPS_ENCODING_SCHEME_CATBOOST,
    "lgbm": HPS_ENCODING_SCHEME_LGBM,
    "es_lgbm": HPS_ENCODING_SCHEME_LGBM,
    "tabpfn": HPS_ENCODING_SCHEME_TABPFN,
    "realmlp": HPS_ENCODING_SCHEME_REALMLP,
    "tabm": HPS_ENCODING_SCHEME_TABM
}


def get_encoding_scheme(estimator: TunableEstimatorType) -> list:
    '''
    Get a deepcopy of the encoding scheme of the HP feature space designed for the input estimator.
    The encoding scheme consists in a ordered list of sklearn transformers to insert in a Pipeline object.
    '''
    return deepcopy(HPS_ENCODING_SCHEME[estimator])