from __future__ import annotations

import warnings
import optuna
from typing import TYPE_CHECKING, Literal, Callable
from sklearn.utils.validation import check_is_fitted
from pytabkit import TabM_D_Classifier
from metatab.utils.core import learn_sklearn_features_attributes, check_predict_features

if TYPE_CHECKING:
    import numpy as np
    from metatab.utils.types import XType, YType



def suppress_tabm_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", 
                message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
                category=FutureWarning
            )
            warnings.filterwarnings(
                action="ignore",
                message="The.*feature has just two bin edges, which means only one bin.*",
                category=UserWarning
            )
            return func(*args, **kwargs)
    return wrapper


# README: 
# The interface does not follow the sklearn design using **params in init.
# For this reason we miss some sklearn features and compabilities.
# These should not be used/useful to us (the useful ones are explicitely implemented). 
class TabMClassifier:
    '''
    Wrapper of pytabkit "TabM_D_Classifier" that allows for:
    1. "auto" batch size option.
    2. Suppression of undesired warnings.
    3. fit method with eval_set-like interface.
    4. learns "feature_names_in_" attribute.
    5. check on fit features at predict level 

    Parameters:
        Accept all TabM_D_Classifier parameters.
    '''
    def __init__(self, batch_size: int | Literal["auto"] = 256, **params):
        self.batch_size=batch_size
        self.params=params


    @suppress_tabm_warnings
    def fit(self, X: XType, y: YType, eval_set: list[tuple[XType, YType]], **kwargs) -> "TabMClassifier":
        '''
        Fit interface accepting the X and y validation sets via "eval_set-like" parameter.
        Accepts the additional tabm args via kwargs.

        Parameters:
            X (XType): data to fit.
            y (YType): data labels to fit.
            eval_set (list[tuple[XType, YType]]):
                List of a SINGLE binary tuple with the Xy validation sets.
                Here we use this signature to be consistent with the GBDTs
                interfaces which accept multiple (potentially) validation sets.
                Cannot be None since we do NOT want to rely on the internal automatic
                system to infer the validation sets. Infact they are always used by TabM_D_Classifier.
            Kwargs:
                Additional keyword parameters accepted by the fit method of TabM_D_Classifier.
                The arguments that pass/specify the validation sets must not be used in favor of
                the `eval_set` parameter.
        '''

        self.batch_size_ = self.batch_size \
            if isinstance(self.batch_size, int) \
            else self._infer_batch_size_from_data(X)
        X_val, y_val = eval_set[0][0], eval_set[0][1]
        self.classifier_ = TabM_D_Classifier(batch_size=self.batch_size_, **self.params)
        self.classifier_.fit(X, y, X_val=X_val, y_val=y_val, **kwargs)
        for k, v in learn_sklearn_features_attributes(X).items(): setattr(self, k, v)
        return self
    

    @suppress_tabm_warnings
    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        check_predict_features(self, X)
        return self.classifier_.predict(X)
    

    @suppress_tabm_warnings
    def predict_proba(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        check_predict_features(self, X)
        return self.classifier_.predict_proba(X)
    

    def set_params(self, **params) -> "TabMClassifier":
        if params.get("batch_size", None) is not None:
            self.batch_size = params.pop("batch_size")
        # here we must update to not lose all previous info
        self.params.update(params)
        return self
    

    def get_params(self, *args, **kwargs) -> dict:
        return {"batch_size": self.batch_size, **self.params}


    @staticmethod
    def _infer_batch_size_from_data(X: XType) -> int:
        n_samples = X.shape[0]
        if n_samples <= 256*3:
            return int(n_samples / 3)
        else:
            # the default
            return 256



def _tabm_sampler_function(trial: optuna.Trial) -> dict:
    '''
    We use the autogluon/tabarena space with minor modifications.
    "https://github.com/autogluon/tabarena/blob/main/tabarena/tabarena/models/tabm/generate.py"
    '''
    point = {
        "arch_type": trial.suggest_categorical("tabm__arch_type", ["tabm", "tabm-mini"]),
        "num_emb_n_bins": trial.suggest_int("tabm__num_emb_n_bins", 2, 128, step=2),
        "d_embedding": trial.suggest_int("tabm__d_embedding", 8, 24, step=4), # high increase in time and memory peak
        "batch_size": trial.suggest_categorical("tabm__batch_size", ["auto", 256]),
        "lr": trial.suggest_float("tabm__lr", 1e-4, 3e-3, log=True),
        # weight_decay can be 0 or loguniform positive value
        "weight_decay": trial.suggest_categorical("tabm__weight_decay", ["zero", "positive"]),
        "d_block": trial.suggest_int("tabm__d_block", 128, 768, step=32), # high increase in time and memory peak
        "n_blocks": trial.suggest_int("tabm__n_blocks", 2, 5), # high increase in time
        # dropout can be 0 or uniform positive value
        "dropout": trial.suggest_categorical("tabm__dropout", ["zero", "positive"]),
        # none, tabm default and realmlp default preprocessing
        "tfms": trial.suggest_categorical("tabm__tfms", [[], ["quantile_tabr"], ["median_center", "robust_scale", "smooth_clip"]])
    }
    
    # resolve conditional weight_decay
    point["weight_decay"] = 0.0 \
        if point["weight_decay"] == "zero" \
        else trial.suggest_float("tabm__pos_weight_decay", 1e-4, 1e-1, log=True)
    
    # resolve conditional dropout
    point["dropout"] = 0.0 \
        if point["dropout"] == "zero" \
        else trial.suggest_float("tabm__pos_dropout", 0.0, 0.5)
    
    return point


class TabMSpec:
    type_classifier = "tabm"
    classifier_class = TabMClassifier
    early_stop_on_validation_set = True
    random_state_parameter = "random_state"
    n_threads_parameter = "n_threads"
    device_parameter = "device"
    main_device = "cuda"
    supported_devices = ["cpu", "cuda"]
    default_preprocessing = "base"
    default_params = {
        "val_metric_name": "cross_entropy",
        "arch_type": "tabm",
        # we avoid quantile transformation since it should have little effect/benefit for our data
        "tfms": [],
        # we increase the patience since epochs with small data are made of few steps
        "patience": 128,
        # we set this explictely not relying on auto which depends on whether embeddings are used
        "n_blocks": 2,
        # we differ from pytabkit using gradient clipping
        "gradient_clipping_norm": 1,
        # in tabm paper it shown that using same or different batches lead to no differences in performance
        # however using the same batch uses less ram
        "share_training_batches": True,
        # mixed precision should speed-up training on GPU
        "allow_amp": True
    }
    fixed_params = {
        "num_emb_type": "pwl",
        "val_metric_name": "cross_entropy",
        "patience": 128,
        "gradient_clipping_norm": 1,
        "share_training_batches": True,
        "allow_amp": True
    }
    callbacks_on_params = None
    hps_sampler_function = _tabm_sampler_function
    initialize_search_function = lambda: None
    set_params_function: Callable[[TabMClassifier, dict], TabMClassifier] = lambda cls, hps: cls.set_params(**hps)
    params_as_object_columns_in_df_search = None