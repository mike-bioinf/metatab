from metatab.estimators.params.space import TuningParams
from metatab.estimators.utils.types import TunableEstimatorType


def pick_estimator_tune_space(estimator: TunableEstimatorType) -> dict:
     match (estimator):
        case ("random_forest"):
            return TuningParams.RF
        case("extra_trees"):
            return TuningParams.EXTRA_TREES
        case ("xgb" | "es_xgb"):
            return TuningParams.XGB
        case ("catboost" | "es_catboost"):
            return TuningParams.CATBOOST
        case ("lgbm" | "es_lgbm"):
            return TuningParams.LGMB
        case ("tabpfn"):
            return TuningParams.TABPFN
        case ("realmlp"):
            return TuningParams.REALMLP
        case ("tabm"):
            return TuningParams.TABM
        case _:
            raise ValueError(f"Unsupported estimator: {estimator}.")