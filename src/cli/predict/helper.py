from estimators.estimators import (
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
    MyLGBMClassifier,
    MyESLGBMClassifier,
    MyTunedLGBMClassifier,
    MyTunedESLGBMClassifier,
    MyTabPFNClassifier,
    MyTunedTabPFNClassifier,
    # MyAutoTabPFNClassifier,
    # MyAesFineTunedTabPFNClassifier
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
            MyLGBMClassifier,
            MyESLGBMClassifier,
            MyTunedLGBMClassifier,
            MyTunedESLGBMClassifier,
            MyTabPFNClassifier,
            MyTunedTabPFNClassifier,
            # MyAutoTabPFNClassifier,
            # MyAesFineTunedTabPFNClassifier
        )
    ):
        raise TypeError("The deserialized object is not of an estimator instance.")
    


def check_estimator_is_fitted(obj) -> None:
    '''Check whether the estimator is fitted'''
    if not hasattr(obj, "estimator_"):
        raise ValueError(
            "The object does not contain a fitted estimator ('estimator_' attribute)."
        )
