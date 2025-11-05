import sys
from typing import Literal, Callable
from copy import deepcopy
from tabpfn.model_loading import _user_cache_dir
from metatab_utils.general import enlist
from estimators.types import TUNABLE_ESTIMATOR_TYPE




def add_root_path_to_tabfpn_ckpt(point: dict) -> dict:
    ckpt = point["model_path"]
    complete_path = _user_cache_dir(sys.platform, appname="tabpfn").resolve() / ckpt
    point["model_path"] = str(complete_path)
    return point


# We define here a set of functions for each estimator that accept as only input the point,
# and return it corrected. The function must/can apply the changes in place.
# The keys of the estimator inner dict define the corrections, in the sense that these
# names are the one accepted and recognized by the 'estimator_corrections' argument 
# of the 'correct_point' method of the PointCorrector class.
ESTIMATOR_SUPPORTED_CORRECTIONS: dict[str, dict[str, Callable[[dict], dict]]] = {
    "tabpfn": {
        "model_path": add_root_path_to_tabfpn_ckpt
    },
    "random_forest": {},
    "catboost": {},
    "es_catboost": {},
    "xgb": {},
    "es_xgb": {},
    "lgbm": {},
    "es_lgbm": {}
}



class PointCorrector:
    '''
    Utility class to apply corrections to hyperparameter points sampled during tuning.

    This class supports two levels of correction:
    (1) General Hyperopt corrections
    (2) Estimator-specific corrections

    All corrections are applied to a deep copy of the input dictionary.
    Even when no corrections are applied, a copy is returned.
    '''
    def correct_point(
        self,
        point: dict,
        apply_hypeopt_corrections: bool = False,
        estimator: TUNABLE_ESTIMATOR_TYPE | None = None,
        estimator_corrections: str | list[str] | Literal["all"] | None = None
    ):
        '''
        Apply the specified corrections to a hyperparameter point.       
        THe hyperopt corrections are always applied first.

        Parameters:
            point (dict): 
                HPs point on which the corrections are applied.
            
            apply_hypeopt_corrections (bool, optional): 
                Whether to apply the hyperopt general corrections to the point.
            
            estimator: (TUNABLE_ESTIMATOR_TYPE | None, optional):
                The type of estimator to which the point refers.
                This info is needed to select the right set of corrections.
            
            estimator_corrections (str | list[str] | Literal["all"] | None, optional):
                Specifies which estimator-specific corrections to apply.
                They are taken from a pre-defined map.
                - "all": apply all supported corrections for the given estimator.
                - str: name of the single correction.
                - list[str]: list of correction names.
                - None: no corrections is applied.

        Returns:
            dict: A corrected copy of the point.
        '''
        self._check_ambiguous_estimator_setting(estimator, estimator_corrections)
        
        # the changes are applied on the copy
        point = deepcopy(point)
        
        if apply_hypeopt_corrections:
            point = self._apply_hyperopt_corrections(point)
         
        # apply estimator corrections
        if estimator is not None:
            selected_estimator_corrections = ESTIMATOR_SUPPORTED_CORRECTIONS[estimator]
            
            # select the desired corrections
            if estimator_corrections != "all":
                estimator_corrections = enlist(estimator_corrections)

                # check for not supported corrections
                for correction in estimator_corrections:
                    if correction not in selected_estimator_corrections.keys():
                        raise KeyError(
                            f"'{correction}' is not a pre-defined correction for '{estimator}' estimator."
                        )

                selected_estimator_corrections = {
                    k:v
                    for k, v in selected_estimator_corrections.items()
                    if k in estimator_corrections
                }

            # apply corrections
            for correct_func in selected_estimator_corrections.values():
                point = correct_func(point)

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
            "inference_config__PREPROCESS_TRANSFORMS"
        ]
        
        for param_to_convert in tuple_to_list_parameters:
            if param_to_convert in point.keys():
                point[param_to_convert] = list(point[param_to_convert])

        return point


    @staticmethod
    def _check_ambiguous_estimator_setting(estimator, estimator_corrections) -> None:
        '''Validate that estimator-related parameters are consistent'''
        if estimator is None and estimator_corrections is not None:
            raise ValueError(
                "Ambiguous estimator-related settings. `estimator` is None but `estimator_corrections` are set."
            )
        if estimator is not None and estimator_corrections is None:
            raise ValueError(
                "Ambiguous estimator-related settings. `estimator_corrections` is None but `estimator` is set."
            )