from sklearn.datasets import make_regression
from _paper.hp_metalearning.surrogate_rf import SurrogateRandomForestRegressor



def test_surrogate_random_forest_predictions():
    X, y = make_regression(random_state=0, n_samples=100)
    srf = SurrogateRandomForestRegressor(n_estimators=10)
    pred_values, pred_uncertainty = srf.fit(X, y).predict(X)
    assert pred_values.size == pred_uncertainty.size, "Predicted values and uncertainty do not have the same size."
    assert pred_values.size == 100,  "The surrogate model returns a wrong number of predictions."