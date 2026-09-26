from __future__ import annotations

import sys
from typing import Callable, TYPE_CHECKING
from copy import deepcopy
from tabpfn.model_loading import _user_cache_dir

if TYPE_CHECKING:
    from metatab.estimators.utils.types import TunableEstimatorType



def add_root_path_to_tabfpn_ckpt(point: dict) -> dict:
    ckpt = point["model_path"]
    complete_path = _user_cache_dir(sys.platform, appname="tabpfn").resolve() / ckpt
    point["model_path"] = str(complete_path)
    return point


# We define here for each estimator needing corrections, a set of functions that accept 
# as only input the point and return it corrected. The function can apply the changes in place.
# The keys of the estimator inner dict define the corrections, in the sense that these
# names are the one accepted and recognized by the 'estimator_corrections' argument 
# of the 'correct_point' method of the PointCorrector class.
ESTIMATOR_SUPPORTED_CORRECTIONS: dict[str, dict[str, Callable[[dict], dict]]] = {
    "tabpfn": {"model_path": add_root_path_to_tabfpn_ckpt}
}


class PointCorrector:
    '''
    Utility class to apply corrections to hyperparameter points sampled during tuning.

    This class supports two levels of correction:
    (1) General Hyperopt corrections
    (2) Estimator-specific corrections

    All corrections are applied to a deep copy of the input dictionary.
    Even when no corrections are applied, a copy is returned.

    Parameters:
        apply_hypeopt_corrections (bool, optional): 
            Whether to apply the hyperopt general corrections to the point.
        
        estimator (TunableEstimatorType | None, optional):
            Needed to select the right set of corrections.
            If None no estimator-specific corrections are applied.
    '''
    def __init__(
        self,
        apply_hypeopt_corrections: bool = False,
        estimator: TunableEstimatorType | None = None
    ):
        self.apply_hypeopt_corrections=apply_hypeopt_corrections
        self.estimator=estimator

    
    def correct_point(self, point: dict):
        '''
        Apply corrections to HP point.       
        The hyperopt corrections are always applied first.

        Parameters:
            point (dict): 
                HP point on which apply the corrections.

        Returns:
            dict: The corrected point. Returns always a deepcopy.
        '''        
        # we correct the copy
        point = deepcopy(point)
        
        if self.apply_hypeopt_corrections:
            point = self._apply_hyperopt_corrections(point)
     
        # apply estimator corrections
        if self.estimator and self.estimator in ESTIMATOR_SUPPORTED_CORRECTIONS.keys():
            for _, correction_func in ESTIMATOR_SUPPORTED_CORRECTIONS[self.estimator].items():
                point = correction_func(point)

        return point

    
    @staticmethod
    def _apply_hyperopt_corrections(point: dict) -> dict:
        '''
        Apply general hyperopt level correction to the sampled params.
        These corrections come from specific quirks of hyperopt.
        The corrections are applied in place.

        In particular the following aspects are addressed:
        - automatic conversion of sampled list to tuple. 
            To distinguish between original and converted tuple we cast 
            the specific parameters explicitly.
        '''
        tuple_to_list_parameters = [
            "inference_config__PREPROCESS_TRANSFORMS",
            "tfms" # for all pytabkit classifiers
        ]
        
        for param_to_convert in tuple_to_list_parameters:
            if param_to_convert in point.keys():
                point[param_to_convert] = list(point[param_to_convert])

        return point
