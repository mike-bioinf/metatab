from sklearn.datasets import load_iris
from pathlib import Path

from fit.estimators import (
    MyRandomizedXGBClassifier, 
    MyESRandomizedXGBClassifier,
    MyXGBClassifier
)

from .test_constants import (
    TEST_ES_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS,
    TEST_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS
)


model_folder = Path(__file__).parents[0] / "fitted_models"
X, y = load_iris(return_X_y=True, as_frame=True)



model_path = model_folder / "my_es_randomized_xgb_classifier.pkl"
if not model_path.exists():
    mer_xgb = MyESRandomizedXGBClassifier(
        preprocessing="base", 
        seed=0,
        fixed_params=TEST_ES_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS
    )
    mer_xgb.fit(X, y).save(model_folder / "my_es_randomized_xgb_classifier.pkl")



model_path = model_folder / "my_randomized_xgb_classifier.pkl"
if not model_path.exists():
    mr_xgb = MyRandomizedXGBClassifier(
        preprocessing="base", 
        seed=0,
        fixed_params=TEST_RANDOMIZED_XGBCLASSIFIER_FIXED_PARAMS
    )
    mr_xgb.fit(X, y).save(model_folder / "my_randomized_xgb_classifier.pkl")



model_path = model_folder / "my_xgb_classifier.pkl"
if not model_path.exists():
    my_xgb = MyXGBClassifier(preprocessing="base", seed=0)
    my_xgb.fit(X, y).save(model_folder / "my_xgb_classifier.pkl")
