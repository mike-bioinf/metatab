import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
    


## This is testing more the external classifier APIs but is for ensure safety
@pytest.mark.parametrize(
    "classifier_class", 
    [RandomForestClassifier, XGBClassifier, CatBoostClassifier, LGBMClassifier, TabPFNClassifier]
)
def test_that_set_params_into_clf_works_for_all_classifiers(classifier_class):
    cls = classifier_class()
    # we set the "random_state" parameter since is shared by all clfs
    cls.set_params(random_state=1234)
    assert cls.get_params()["random_state"] == 1234, "set_params is not properly implemented for the classifier"