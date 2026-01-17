import pytest
import json
from sklearn.datasets import make_regression
from metatab.metalearning.surrogate_rf import SurrogateRandomForestRegressor
from metatab.metalearning.load import resolve_surrogate_models_folder



def test_surrogate_random_forest_predictions():
    X, y = make_regression(random_state=0, n_samples=100)
    srf = SurrogateRandomForestRegressor(n_estimators=10)
    pred_values, pred_uncertainty = srf.fit(X, y).predict(X)
    assert pred_values.size == pred_uncertainty.size, "Predicted values and uncertainty do not have the same size."
    assert pred_values.size == 100,  "The surrogate model returns a wrong number of predictions."



### --- testing the surrogate models loading process ----------------------------------------------------------------- 

TEST_MANIFEST = {
    "0.1": {
        "package_versions": ">=0.1,<0.4",
        "models_subpackage": "models_0.1"
    },
    "0.4": {
        "package_versions": ">=0.4,<1.0",
        "models_subpackage": "models_0.4"
    }
}


@pytest.mark.parametrize(
    "pkg_version,expected_folder",
    [
        ("0.1.0", "models_0.1"),
        ("0.3.9", "models_0.1"),
        ("0.4.0", "models_0.4"),
        ("0.9.9", "models_0.4")
    ]
)
def test_resolve_surrogate_models_folder_utility(
    tmp_path, monkeypatch, pkg_version, expected_folder
):

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(TEST_MANIFEST))

    monkeypatch.setattr(
        "metatab.metalearning.load.hf_hub_download",
        lambda *args, **kwargs: str(manifest_path)
    )

    monkeypatch.setattr(
        "metatab.metalearning.load.version",
        lambda *args, **kwargs: pkg_version
    )

    assert resolve_surrogate_models_folder() == expected_folder