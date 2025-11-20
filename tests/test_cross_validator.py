import numpy as np
import pandas as pd
from hyperopt.pyll.stochastic import sample
from hp_search.cv import CrossValidator
from lightgbm import LGBMClassifier
from estimators.params import TuningParams
from estimators.estimators.lgbm import ignore_lgbm_feature_name_warning
from sklearn.datasets import load_iris



def create_cross_validator() -> CrossValidator:
    lgbm_fixed_params = TuningParams.LGBM_FIXED_PARAMS
    clf = LGBMClassifier(**lgbm_fixed_params)

    cross_validator = CrossValidator(
        clf_or_pipe=clf,
        clf_random_state_parameter="random_state",
        early_stop_on_validation_set=False,
        eval_set_parameter=None,
        validation_set_size=None,
        fit_classifier_kwargs={},
        metric="logloss",
        n_folds=3,
        n_repeats=1,
        seed=0
    )

    return cross_validator


@ignore_lgbm_feature_name_warning
def fit_cross_validator(cross_validator: CrossValidator) -> tuple[float, pd.DataFrame, CrossValidator]:
    X, y = load_iris(return_X_y=True, as_frame=False)
    lgbm_tune_space = TuningParams.LGMB_C0
    params = sample(lgbm_tune_space, rng=np.random.default_rng(0))
    loss, df_info = cross_validator.fit(X, y, params, "sum", True)
    return loss, df_info, cross_validator


def test_cross_validator_fitting_procedure_not_raise_expections():
    _ = fit_cross_validator(create_cross_validator())


def test_cross_validator_works_as_expected():
    _, df_info, _ = fit_cross_validator(create_cross_validator())
    assert df_info.shape[0] == 3, "Wrong number of rows for df_info"

    try:
        for col in ["repeat", "fold", "loss"]:
            if col not in df_info.columns:
                raise ValueError("")
    except:
        assert False, f"{col} not found in df_info"


def test_reproducibility_of_cross_validator_results():
    _, df_info, _ = fit_cross_validator(create_cross_validator())
    _, df_info_2, _ = fit_cross_validator(create_cross_validator())
    assert np.allclose(df_info["loss"].to_numpy(), df_info_2["loss"].to_numpy(), atol=1e-6, rtol=0), "CrossValidator lacks of reproducibility."