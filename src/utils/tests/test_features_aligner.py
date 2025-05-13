import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from utils.features_aligner import FeaturesAligner



def test_features_aligner_reindex_dataframe():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_test = X[["age", "sex"]]
    fa = FeaturesAligner()
    X_test_trasformed = fa.fit(X).transform(X_test)
    assert X_test_trasformed.columns.equals(X.columns), "FeaturesAligner does not align the feature spaces"



def test_feature_aligner_skip_transformation_with_train_numpy_arrays():
    X, y = load_diabetes(return_X_y=True, as_frame=False)
    X_test = X[:, 0:3]
    fa = FeaturesAligner()
    X_test_transformed = fa.fit(X).transform(X_test)
    assert X_test_transformed.shape[1] == 3, "FeaturesAligner does not skip the transformation when not fitted with pandas DataFrame" 



def test_features_aligner_convertion_mechanisms_when_skipping_trasformation():
    X, y = load_diabetes(return_X_y=True, as_frame=False)
    X_test = X[:, 0:3]
    
    fa = FeaturesAligner(convert_on_skip=True)
    fa1 = FeaturesAligner(convert_on_skip=False)
    fa2 = FeaturesAligner(convert_on_skip=False)
    fa2.set_output(transform="pandas")

    assert isinstance(fa.fit(X).transform(X_test), pd.DataFrame), "FeaturesAligner does not convert the numpy array in input to transform to a pd DataFrame"
    assert isinstance(fa1.fit(X).transform(X_test), np.ndarray), "FeaturesAligner converts the array in input to transform into a DataFrame when convert_on_skip is False"
    assert isinstance(fa2.fit(X).transform(X_test), pd.DataFrame), "set_output method does not ovverride the convert_on_skip instruction"
