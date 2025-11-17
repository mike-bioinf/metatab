from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from estimators.utils.fit import fit_with_early_stop_on_validation_set



def test_that_pipeline_is_fitted_after_completion():
    X, y = make_classification()
    
    pipe = make_pipeline(
        VarianceThreshold(), 
        LGBMClassifier(
            objective="binary", 
            min_child_samples=1,
            metric="binary_logloss",
            verbose=-1,
            random_state=0
        )
    )

    _ = fit_with_early_stop_on_validation_set(
        clf_or_pipe=pipe,
        X=X,
        y=y,
        seed=0,
        validation_set_size=0.3,
        eval_set_parameter="eval_set",
        fit_classifier_kwargs={},
        return_fit_time=False
    )
    
    assert pipe[-1].fitted_, "The pipeline is not fitted in place"
