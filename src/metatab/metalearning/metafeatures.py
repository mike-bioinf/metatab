import re
import numpy as np
from typing import Iterable, Any
from sklearn.utils.validation import check_is_fitted
from pymfe.mfe import MFE
from metatab.utils.general import ensure_or_create, enlist
from metatab.utils.types import XType, YType



HIGH_SPARSITY_THRESHOLD = 0.8
LOW_SPARSITY_THRESHOLD = 0.2


def compute_fraction_full_zero_columns(X: np.ndarray) -> float:
    return np.all((X == 0), axis=0).sum() / X.shape[1]


def compute_fraction_full_dense_columns(X: np.ndarray) -> float:
    return (_get_columns_density_scores(X) == 1).sum() / X.shape[1]


def compute_fraction_high_sparsity_columns(X: np.ndarray) -> float:
    return ((1 - _get_columns_density_scores(X)) >= HIGH_SPARSITY_THRESHOLD).sum() / X.shape[1] 


def compute_fraction_low_sparsity_columns(X: np.ndarray) -> float:
    return ((1 - _get_columns_density_scores(X)) <= LOW_SPARSITY_THRESHOLD).sum() / X.shape[1]


def _get_columns_density_scores(X: np.ndarray) -> np.ndarray:
    return np.mean((X != 0), axis=0)


MAP_SPARSE_METAFEATURES = {
    "fraction_full_zero_columns": compute_fraction_full_zero_columns,
    "fraction_full_dense_columns": compute_fraction_full_dense_columns,
    "fraction_high_sparse_columns": compute_fraction_high_sparsity_columns,
    "fraction_low_sparse_columns": compute_fraction_low_sparsity_columns
}


# MFE distinguish between "nan" and "base" version for all the summary funcs,
# with the nan version that is able to filter out the nan values in its computation.
# In our case the 2 versions give the same results generally. 
# This has been evaluated on the original cohort of 32 datasets,
# and therefore we keep only the base ones to avoid duplicating the metafeatures.
# We keep the base and not the nan version since our surrogate model (randomforest)
# is able to handle and learn from this information. So we capture this info here.
BASE_SUMMARY = [
    "mean",
    "sd",
    "var",
    "count",
    "histogram",
    "iq_range",
    "kurtosis",
    "max",
    "median",
    "min",
    "quantiles",
    "range",
    "skewness",
    "sum",
    "powersum",
    "pnorm"
]



class CustomMFE:
    '''
    Custom version of the `MFE` class of `pymfe` package.

    Changes:
    - We can extract the metafeatures from pandas DataFrame and Series.
    - We can compute additional sparsity metafeatures. These are simple metafeatures 
    on which the summarization function are not applied. They can be computed on the
    original data only (if transformation want to be applied an error is raised).
    - The "landmarking" group is suppressed since it is equivalent to "relative".
    The relative versions are selected since more robust in our scenario where 
    we collect the info on multiple datasets with different difficulty.
    - We implement only the "base" `extract` utility.
    - Different default values for `groups`, `summary`, `score` and `suppress_warnings` parameters.

    The class accepts the same set of parameters of the MFE class, with the following expections:
    - We can specify the `additional_sparse` group in `groups` parameter.
    - We can specify the MAP_SPARSE_METAFEATURES keys in `features` parameter.
    - `random_state` is renamed as `seed` with a default value of 42.
    '''
    def __init__(
        self,
        groups: str | Iterable[str] = "all",
        features: str | Iterable[str] = "all",
        summary: str | Iterable[str] = BASE_SUMMARY,
        measure_time: str | None = None,
        wildcard: str = "all",
        score: str = "balanced-accuracy",
        num_cv_folds: int = 10,
        shuffle_cv_folds: bool = False,
        lm_sample_frac: float = 1,
        hypparam_model_dt: dict[str, Any] | None = None,
        suppress_warnings: bool = True,
        seed: int = 42
    ):
        self.map_group_metafeature = self._create_map_group_metafeature()
        cleaned_groups = self._clean_groups_input_from_landmarking(groups)
        cleaned_groups, cleaned_features = self._handle_additional_sparse_options(cleaned_groups, features)

        if cleaned_groups:
            self.mfe = MFE(
                groups=cleaned_groups,
                features=cleaned_features,
                summary=summary,
                measure_time=measure_time,
                wildcard=wildcard,
                score=score,
                num_cv_folds=num_cv_folds,
                shuffle_cv_folds=shuffle_cv_folds,
                lm_sample_frac=lm_sample_frac,
                hypparam_model_dt=hypparam_model_dt,
                suppress_warnings=suppress_warnings,
                random_state=seed
            )
        else:
            self.mfe = None


    @staticmethod
    def _clean_groups_input_from_landmarking(groups: str | list[str]) -> list[str]:
        '''
        Here we take care of the "landmarking" group suppression,
        and expand the wildcard to the groups which they refer.
        '''
        if isinstance(groups, str):
            if groups == "all":
                return  [g for g in MFE().valid_groups() if g != "landmarking"] + ["additional_sparse"]
            elif groups == "default":
                return ["general", "info-theory", "statistical", "model-based", "relative"] + ["additional_sparse"]
            else:
                return enlist(groups)
        
        else:
            if "landmarking" in groups:
                raise ValueError(
                    "'landmarking' group is not valid. Require the 'relative' group instead."
                )    
            return groups


    def _handle_additional_sparse_options(self, groups, features) -> tuple[list[str], list[str]]:
        '''
        Handles the additional sparse and features options.
        - If `additional_sparse` or `all` is in groups, we allow sparse metafeatures.
        - If sparse `groups` are set but not sparse metafeatures are selected then raises an error.
        - Sets the `additional_sparse_metafeatures` attribute, the list of the additional metafeatures to compute (can be empty).
        - Returns "cleaned" `groups` and `features` for MFE class.
        '''
        features = enlist(features)
        
        is_sparse_group_requested = any([g == "additional_sparse" for g in groups])

        are_sparse_metafeatures_requested = (
            any([sparse_feature in features for sparse_feature in MAP_SPARSE_METAFEATURES.keys()]) or
            "all" in features
        )
   
        # raise an error if sparse groups are specified but no sparse metafeature
        # you must not check the opposite since the features scope is set by groups
        if is_sparse_group_requested and not are_sparse_metafeatures_requested:
            raise ValueError(
                "Sparse additional metafeatures requested with missing compatible 'groups' specification."
            )
        
        # the scope of features is set by groups,
        # so if not sparse groups then no sparse features even with "all"
        if is_sparse_group_requested:
            if "all" in features:
                self.additional_sparse_metafeatures = list(MAP_SPARSE_METAFEATURES.keys())
            else:
                self.additional_sparse_metafeatures = [f for f in MAP_SPARSE_METAFEATURES.keys() if f in features]
        else:
            self.additional_sparse_metafeatures = []
        
        cleaned_groups = [g for g in groups if g != "additional_sparse"]
        cleaned_features = [f for f in features if f not in MAP_SPARSE_METAFEATURES.keys()]

        return cleaned_groups, cleaned_features
    

    def fit(
        self, 
        X: XType, 
        y: YType,
        transform_num: bool = False,
        transform_cat: str = None,
        rescale: str | None = None,
        rescale_args: dict[str, Any] | None = None,
        cat_cols: str | Iterable[int] | None = None,
        check_bool: bool = False,
        precomp_groups: str | None = "all",
        wildcard: str = "all",
        suppress_warnings: bool = True,
        verbose: int = 0,
        **kwargs
    ) -> "CustomMFE":
        '''
        Take the same set of parameters of the MFE class "fit" method.
        
        The only differences are:
        - this method supports only pandas X and y inputs
        - different default values for the `transform_num`, `transform_cat`, 
        `rescale`, `cat_cols` and `suppress_warnings` parameters.
        
        See here for additional details: 
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.mfe.MFE.html#pymfe.mfe.MFE".
        '''
        # here we not check for numerical to categorical transformation 
        # since the added sparse metafetures should not be applied on those.
        mfe_does_transformation = rescale is not None or transform_cat is not None
        
        if mfe_does_transformation and self.additional_sparse_metafeatures:
            raise ValueError(
                "Is not possible to compute additional_sparse metafeature on transformed data."
            )
        
        self._X = X if isinstance(X, np.ndarray) else X.to_numpy()
        self._y = y if isinstance(y, np.ndarray) else y.to_numpy()
        
        if self.mfe:
            _ = self.mfe.fit(
                self._X, 
                self._y,
                transform_num=transform_num,
                transform_cat=transform_cat,
                rescale=rescale,
                rescale_args=rescale_args,
                cat_cols=cat_cols,
                check_bool=check_bool,
                precomp_groups=precomp_groups,
                wildcard=wildcard,
                suppress_warnings=suppress_warnings,
                verbose=verbose, 
                **kwargs
            )
        
        self.is_fitted_ = True
        return self
    
    
    def extract(
        self,
        verbose: int = 0,
        enable_parallel: bool = False,
        suppress_warnings: bool = True,
        add_features: None | dict = None,
        **kwargs
    ) -> tuple[dict, list]:
        '''
        Extract metafeatures.
        Take the same set of parameters of the MFE class extract method with the expections:
        - `suppress_warnings` is set by default to True.
        - `add_features` is a new parameter that allows to add metafeatures that do not depend on data. 
        - `out_type` parameter is removed since internally fixed to dict.
        
        Returns:
          tuple[dict,list]: A binary tuple with:
          1. The extracted metafeatures in a dict name:value.
          2. The group info indicating the metafeatures original groups. 
        '''
        check_is_fitted(self, "is_fitted_")
        add_features = ensure_or_create(add_features, dict)
        out = {}

        if self.mfe:
            out = self.mfe.extract(
                verbose=verbose, 
                enable_parallel=enable_parallel, 
                suppress_warnings=suppress_warnings, 
                out_type=dict, 
                **kwargs
            )

            # the pymfe dict output is divided in mtf_names and mtf_vals lists
            out = {
                out["mtf_names"][i]: out["mtf_vals"][i]
                for i in range(len(out["mtf_names"]))
            }

        sparse_out = {}
        for smtf in self.additional_sparse_metafeatures:
            sparse_out[smtf] = MAP_SPARSE_METAFEATURES[smtf](self._X)
        
        out = {**out, **sparse_out}

        # extract info about original groups
        original_groups = []

        for mft in out.keys():
            # removing the summary func part to avoid potential erroneous matching
            # for "relative" metafeatures only the root is needed for group attribution
            mft = re.sub("\\..*", "", mft)
            found = False
            for group, reference_mtfs in self.map_group_metafeature.items():
                for reference_mft in reference_mtfs:
                    if re.match(mft, reference_mft):
                        original_groups.append(group)
                        found = True
                        break 
                if found:
                    break

        # check name collision and then add
        if add_features:
            for k in add_features.keys():
                if k in out.keys():
                    raise KeyError(
                        f"The additional feature name '{k}' is colliding with existing metafeatures."
                    )
            out = {**out, **add_features}
            original_groups.extend(["external"] * len(add_features))

        return out, original_groups
    

    @staticmethod
    def _create_map_group_metafeature() -> dict:
        '''
        Creates the group-feature map indicating the 
        group in which the different metafeatures belong.
        '''
        mfe = MFE()
        map = {}

        for group in mfe.valid_groups():
            map[group] = list(mfe.valid_metafeatures(group))

        # remove "landmarking" group which is suppressed in favor of "relative"
        del map["landmarking"]

        # adding additional sparse metafeatures to statistical group
        map["statistical"].extend(MAP_SPARSE_METAFEATURES.keys())
        
        return map