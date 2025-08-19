from estimators import (
    MyRandomForestClassifier,
    MyTunedRandomForestClassifier,
    MyXGBClassifier,
    MyESXGBClassifier,
    MyTunedXGBClassifier,
    MyTunedESXGBClassifier,
    MyCatBoostClassifier,
    MyESCatBoostClassifier,
    MyTunedCatBoostClassifier,
    MyTunedESCatBoostClassifier,
    MyTabPFNClassifier
)



def check_type_deserialized_object(obj) -> None:
    '''Check whether the pickle deserialized object is an estimator instance'''
    if not isinstance(
        obj, 
        (
            MyRandomForestClassifier,
            MyTunedRandomForestClassifier,
            MyXGBClassifier,
            MyESXGBClassifier,
            MyTunedXGBClassifier,
            MyTunedESXGBClassifier,
            MyCatBoostClassifier,
            MyESCatBoostClassifier,
            MyTunedCatBoostClassifier,
            MyTunedESCatBoostClassifier,
            MyTabPFNClassifier
        )
    ):
        raise TypeError("The deserialized object is not of an expected estimator class.")
    


def check_estimator_is_fitted(obj) -> None:
    '''Check whether the estimator is fitted'''
    if not hasattr(obj, "estimator_"):
        raise ValueError(
            "The object does not contain a fitted estimator ('estimator_' attribute)."
        )
