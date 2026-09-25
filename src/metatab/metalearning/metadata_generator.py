from __future__ import annotations

import warnings
import pandas as pd
from typing import TYPE_CHECKING
from sklearn.utils.validation import check_is_fitted
from metatab.metatab_utils.general import ensure_or_create

if TYPE_CHECKING:
    from metatab.metalearning.sampler import HyperoptRandomSampler
    from metatab.metalearning.metafeatures import CustomMFE
    from metatab.hp_search.point_corrector import PointCorrector
    from metatab.metatab_utils.types import XType, YType



class MetadataGenerator():
    '''
    Class that manages the hp sampler, point corrector and metafeature extractor
    to generate metadata from a hp space and some data from which extract metafeatures.

    Parameters:
        sampler (HyperoptRandomSampler):
            Sampler that allows to sample hp points from a space.
        point_corrector (PointCorrector):
            Corrector of the sampled points.
        mfe (CustomMFE):
            CustomMFE to extract data metafeatures.
    '''
    def __init__(
        self,
        sampler: HyperoptRandomSampler,
        point_corrector: PointCorrector,
        mfe: CustomMFE,
    ):
        self.sampler=sampler
        self.point_corrector=point_corrector
        self.mfe=mfe

    
    def fit(
        self, 
        X: XType, 
        y: YType, 
        hp_space: dict,
        seed: int
    ) -> "MetadataGenerator":
        '''
        Initialize the generator with the data, hyperparameter space, and random seed.
        The provided `hp_space` must be compatible with the assigned sampler.

        Parameters:
            X (XType): Feature matrix.
            y (YType): Target vector.
            hp_space (dict): Hyperparameter space.
            seed (int): Random seed controlling candidate sampling.

        Returns:
            MetadataGenerator: The fitted instance.
        '''
        self.X=X
        self.y=y
        self.hp_space=hp_space
        self.seed=seed
        self.is_fitted_=True
        return self
    

    def generate(
        self,
        n_points: int,
        mfe_fit_kwargs: None | dict = None,
        mfe_extract_kwargs: None | dict = None,
        set_metagroups_in_index: bool = False,
        set_as_last: str | None = "preprocessing"
    ) -> tuple[pd.DataFrame, list[dict]]:
        '''
        Generate the meta-data, i.e. sampled hps + data metafeatures.

        Parameters:
            n_points (int): 
                Number of points to draw from the hp space.

            mfe_fit_kwargs (None | dict, optional):
                Kwargs to pass to the mfe `fit` method.
            
            mfe_extract_kwargs (None | dict, optional):
                Kwargs to pass to the mfe `extract` method.
            
            set_metagroups_in_index (bool, optional):
                Whether to set the "group" info in the metadata column index.
                The group info is the level which informs about the group
                in which the hps and metafeatures belong. These groups are
                defined based on the existing literature on metafeatures.
                The hps are put in the group "hps".
                The resulting multiindex has two levels namely "group"
                and "feature" in this order. 

            set_as_last (str | None, optional):
                Set the specified feature as last feature in the resulting dataframe of candidates.
                It is set to "preprocessing" by default since usually needed for this feature.
                If the feature is not present then it does nothing.

        Returns:
            tuple[pd.DataFrame,list[dict]]:
            Returns the meta-data plus the list of hp points used to build it.
            Importantly the meta-data and points order matches, meaning
            that the first row is built upon the first point in the list and so on.
        '''
        check_is_fitted(self, "is_fitted_")
        mfe_fit_kwargs = ensure_or_create(mfe_fit_kwargs, dict)
        mfe_extract_kwargs = ensure_or_create(mfe_extract_kwargs, dict)

        candidate_points = [
            self.point_corrector.correct_point(sample)
            for sample in self.sampler.fit(self.hp_space, self.seed).sample_points(n_points)
        ]
        
        df_candidate_points = pd.DataFrame(candidate_points)
        n_hps = df_candidate_points.shape[1]
        metafeatures, groups = self.mfe.fit(self.X, self.y, **mfe_fit_kwargs).extract(**mfe_extract_kwargs)
        
        # we create a copy since the original df is not optimized in memory due to assign
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning)
            df_candidate_points = df_candidate_points.assign(**metafeatures).copy()
            
        if set_metagroups_in_index:
            groups = ["hps"] * n_hps + groups
            df_candidate_points.columns = pd.MultiIndex.from_arrays(
                [groups, df_candidate_points.columns],
                names=["group", "feature"]
            )

        if set_as_last:
            columns = df_candidate_points.columns

            if isinstance(columns, pd.MultiIndex):
                feature_values = columns.get_level_values("feature")

                if set_as_last in feature_values:
                    mask = feature_values == set_as_last
                    df_candidate_points = df_candidate_points.loc[:, list(columns[~mask]) + list(columns[mask])]

            elif set_as_last in columns:
                df_candidate_points = df_candidate_points[
                    [col for col in columns if col != set_as_last] + [set_as_last]
                ]

        return df_candidate_points, candidate_points