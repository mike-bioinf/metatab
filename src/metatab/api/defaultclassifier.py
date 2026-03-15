from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from metatab.classifiers.registry import get_classifier_specs_from_registry
from metatab.utils.core import fit_with_early_stop_on_validation_set

from metatab.utils.api import (
    create_pipeline, 
    check_validation_set,
    check_validation_set_classifier_combination,
    encode_y, 
    handle_device
)


if TYPE_CHECKING:
    import numpy as np
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.utils.types import XType, YType, DefaultClassifierType ## refactor add type



class DefaultClassifier(ClassifierMixin, BaseEstimator):
    '''
    Refactor: add documenatation for parameters
    '''
    def __init__(
        self,
        type_classifier: DefaultClassifierType,
        preprocessing: PreprocessingStrategy = "estimator_default",
        seed: int = 0,
        n_threads: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ):
        self.type_classifier=type_classifier
        self.preprocessing=preprocessing
        self.seed=seed
        self.n_threads=n_threads
        self.device=device


    def fit(
        self,
        X: XType,
        y: YType,
        validation_set_size: float | None = None
    ):
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        classifier_spec = get_classifier_specs_from_registry(self.type_classifier)

        check_validation_set(validation_set_size)
        check_validation_set_classifier_combination(validation_set_size, classifier_spec, self.type_classifier)
        resolved_device = handle_device(self.device, classifier_spec, self.type_classifier)
        
        resolved_preprocessing = classifier_spec.default_preprocessing \
            if self.preprocessing == "estimator_default"\
            else self.preprocessing
        
        label_encoder, y = encode_y(X, y)

        pipe = create_pipeline(
            classifier_class=classifier_spec.classifier_class,
            classifier_params=classifier_spec.default_params,
            callbacks_on_classifier_params=classifier_spec.callbacks_on_params,
            y=y,
            preprocessing=resolved_preprocessing,
            classifier_random_state_parameter=classifier_spec.random_state_parameter,
            classifier_nthreads_paramater=classifier_spec.n_threads_parameter,
            classifier_device_parameter=classifier_spec.device_parameter,
            seed=self.seed,
            n_threads=self.n_threads,
            device=resolved_device
        )

        if classifier_spec.early_stop_on_validation_set:
            self.estimator_ = fit_with_early_stop_on_validation_set(
                pipe=pipe,
                X=X,
                y=y,
                seed=self.seed,
                validation_set_size=validation_set_size,
            )
        else:
            self.estimator_ = pipe.fit(X, y)

        self.classes_ = label_encoder.classes_
        return self
    

    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
        

    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)