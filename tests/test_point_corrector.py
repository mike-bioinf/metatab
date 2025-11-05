import pytest
import sys
from hp_search.point_corrector import PointCorrector
from tabpfn.model_loading import _user_cache_dir



TABPFN_LIKE_POINT = {
        "inference_config__PREPROCESS_TRANSFORMS": (33, 22),
        "model_path": "filename.ckpt"
    }


def test_that_point_corrector_applies_all_corrections():
    cache_path = _user_cache_dir(sys.platform)
    point_corrector = PointCorrector()
    corrected_point = point_corrector.correct_point(
        TABPFN_LIKE_POINT,
        apply_hypeopt_corrections=True,
        estimator="tabpfn",
        estimator_corrections="all"
    )
    assert corrected_point["model_path"] == str(cache_path / "filename.ckpt"), "The point corrector is not working"
    assert isinstance(corrected_point["inference_config__PREPROCESS_TRANSFORMS"], list), "The point corrector is not working"



def test_that_point_corrector_applies_only_hyperopt_corrections():
    cache_path = _user_cache_dir(sys.platform)
    point_corrector = PointCorrector()
    corrected_point = point_corrector.correct_point(
        TABPFN_LIKE_POINT,
        apply_hypeopt_corrections=False, 
        estimator="tabpfn",
        estimator_corrections="all"
    )
    assert corrected_point["model_path"] == str(cache_path / "filename.ckpt"), "The point corrector is not working"
    assert isinstance(corrected_point["inference_config__PREPROCESS_TRANSFORMS"], tuple), "The point corrector is not working"



def test_that_point_corrector_applies_only_estimator_corrections():
    point_corrector = PointCorrector()
    corrected_point = point_corrector.correct_point(
        TABPFN_LIKE_POINT,
        apply_hypeopt_corrections=True, 
        estimator=None,
        estimator_corrections=None
    )
    assert corrected_point["model_path"] == "filename.ckpt", "The point corrector is not working"
    assert isinstance(corrected_point["inference_config__PREPROCESS_TRANSFORMS"], list), "The point corrector is not working"



def test_that_point_corrector_raise_error_for_unrecognized_corrections():
    point_corrector = PointCorrector()
    with pytest.raises(KeyError, match=".*undefined_correction.*"):
        point_corrector.correct_point(
            TABPFN_LIKE_POINT,
            apply_hypeopt_corrections=False, 
            estimator="tabpfn",
            estimator_corrections="undefined_correction"
        )