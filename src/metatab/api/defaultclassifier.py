from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from metatab.classifiers.registry import get_classifier_specs_from_registry
from metatab.utils.core import fit_using_validation_set
from metatab.utils.pipeline import build_pipeline

from metatab.utils.api import ( 
    check_validation_set,
    check_validation_set_classifier_combination,
    encode_y, 
    handle_device
)


if TYPE_CHECKING:
    import numpy as np
    from metatab.preprocessing.types import PreprocessingStrategy
    from metatab.utils.types import XType, YType, DefaultClassifierType



class DefaultClassifier(ClassifierMixin, BaseEstimator):
    '''
    Run a classifier with the default configuration.

    Parameters:
        type_classifier (DefaultClassifierType):
            Classifier to run.

        preprocessing (PreprocessingStrategy, optional):
            Preprocessing strategy to apply.
        
        seed (int, optional):
            Random seed controlling classifer randomness.
            Ignored by classifiers that do not allow to set this (AutoGluon)

        n_threads (int, optional):
            Number of threads used to parallelize classifier fitting.
            Ignored by classifiers that does not support this.

        device (Literal["cpu", "cuda", "auto"], optional):
            Device to fit the model(s) on.
            - "cpu" or "cuda" explicitly selects the device.
            - "auto" falls on "cuda" if available and supported by the classifier; otherwise "cpu".

    ## Attributes:
        classes_ (np.ndarray): 
            The array of class labels learnt at fit time.
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
    ) -> "DefaultClassifier":
        '''
        Fit the classifier.
        
        Parameters:
            X (XType): Data to fit.
            
            y (Ytype): Data labels to fit.
            
            validation_set_size (None | float, optional): 
                Size of the validation set.
                Must be provided only for classifiers that use the validation set in the fit process.
                An error is raised when it is provided and the classifier does not support it,
                or when it is not provided and the classifier needs it.
        
        Returns:
            self
        '''
        check_X_y(X, y, dtype=None, ensure_all_finite=False)
        classifier_spec = get_classifier_specs_from_registry(self.type_classifier)
        check_validation_set(validation_set_size)
        check_validation_set_classifier_combination(validation_set_size, classifier_spec, self.type_classifier)
        resolved_device = handle_device(self.device, classifier_spec, self.type_classifier)        
        label_encoder, y = encode_y(X, y)

        pipe = build_pipeline(
            preprocessing=self.preprocessing,
            hps=classifier_spec.default_params,
            classifier_spec=classifier_spec,
            classifier_seed=self.seed,
            classifier_device=resolved_device,
            classifier_nthreads=self.n_threads,
            y=y
        )

        if classifier_spec.early_stop_on_validation_set:
            self.estimator_ = fit_using_validation_set(
                pipe=pipe,
                X=X,
                y=y,
                validation_set_size=validation_set_size,
                seed=self.seed
            )
        else:
            self.estimator_ = pipe.fit(X, y)

        self.classes_ = label_encoder.classes_
        return self
    

    def predict(self, X: XType) -> np.ndarray:
        '''
        Predict class for X.

        Parameters:
            X (XType): Input samples.

        Returns:
            np.ndarray: The predicted classes.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
        

    def predict_proba(self, X: XType) -> np.ndarray:
        '''
        Predict class probabilities for X.

        Parameters:
            X (XType): Input samples.
        
        Returns:
            np.ndarray: The class probabilities of the input samples.
        '''
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)