from metatab.estimators.params import TuningParams
from metatab.estimators.utils.types import TunableEstimatorType



# Dict of default tuning spaces for each tunable estimator.
# The default spaces are identified based on our paper preanalysis
DEFAULT_ESTIMATORS_TUNE_SPACES = {
    "random_forest": ("c0", TuningParams.RF_C0),
    "extra_trees": ("c0", TuningParams.EXTRA_TREES_C0),
    "xgb": ("c0", TuningParams.XGB_C0), 
    "es_xgb": ("c0", TuningParams.XGB_C0), 
    "catboost": ("c0", TuningParams.CATBOOST_C0), 
    "es_catboost": ("c0", TuningParams.CATBOOST_C0),
    "lgbm": ("c0", TuningParams.LGMB_C0), 
    "es_lgbm": ("c0", TuningParams.LGMB_C0),
    "tabpfn": ("c0", TuningParams.TABPFN_C0)
}


def pick_estimator_tune_space(estimator: TunableEstimatorType, space: str) -> dict:
     match (estimator, space):
        case ("random_forest", "c0"):
            return TuningParams.RF_C0
         
        case("extra_trees", "c0"):
            return TuningParams.EXTRA_TREES_C0
        
        case ("xgb" | "es_xgb", "c0"):
            return TuningParams.XGB_C0
        case ("xgb" | "es_xgb", "c1"):
            return TuningParams.XGB_C1
        case ("xgb" | "es_xgb", "c2"):
            return TuningParams.XGB_C2
        case ("xgb" | "es_xgb", "c3"):
            return TuningParams.XGB_C3
        case ("xgb" | "es_xgb", "c4"):
            return TuningParams.XGB_C4
        
        case ("catboost" | "es_catboost", "c0"):
            return TuningParams.CATBOOST_C0
        case ("catboost" | "es_catboost", "c1"):
            return TuningParams.CATBOOST_C1
        case ("catboost" | "es_catboost", "c2"):
            return TuningParams.CATBOOST_C2
        case ("catboost" | "es_catboost", "c3"):
            return TuningParams.CATBOOST_C3
        case ("catboost" | "es_catboost", "c4"):
            return TuningParams.CATBOOST_C4
        case ("catboost" | "es_catboost", "c5"):
            return TuningParams.CATBOOST_C5
        case ("catboost" | "es_catboost", "c6"):
            return TuningParams.CATBOOST_C6

        case ("lgbm" | "es_lgbm", "c0"):
            return TuningParams.LGMB_C0
        case ("lgbm" | "es_lgbm", "c1"):
            return TuningParams.LGMB_C1

        case ("tabpfn", "c0"):
            return TuningParams.TABPFN_C0
        
        case (_, "default"):
            return DEFAULT_ESTIMATORS_TUNE_SPACES[estimator][1]
            
        case _:
            raise ValueError(
                f"Unsupported configuration '{space}' for '{estimator}' estimator."
            )