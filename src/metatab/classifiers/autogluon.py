import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from autogluon.tabular import TabularPredictor
from metatab.utils.types import XType, YType



class AutoGluonClassifier:
    def __init__(
        self,
        eval_metric: str,
        path: str,
        presets: str,
        time_limit: int = 60,
        num_cpus: int | str = "auto",
        num_gpus: int | str = "auto",
        auto_stack: bool = True,            
        verbosity: int = 0
    ):
        self.eval_metric=eval_metric
        self.path=path
        self.presets=presets
        self.time_limit=time_limit
        self.num_cpus=num_cpus
        self.num_gpus=num_gpus
        self.auto_stack=auto_stack
        self.verbosity=verbosity

    def fit(self, X: XType, y: YType) -> "AutoGluonClassifier":
        # we have X and y uniformed in type in our program
        if isinstance(X, pd.DataFrame):
            if y.name in X.columns:
                raise KeyError(f"y name '{y.name}' is duplicated in X columns.")
            target_column = y.name
        else:
            X = pd.DataFrame(X)
            y = pd.Series(y)
            X.columns = [f"col_{i}" for i in range(X.shape[1])]
            target_column =  f"col_{X.shape[1]}"

        problem_type = "multiclass" if y.unique().size > 2 else "binary"
        train_data = X.copy()
        train_data[target_column] = y

        tabular_predictor = TabularPredictor(
            label=target_column,
            problem_type=problem_type,
            eval_metric=self.eval_metric,
            path=self.path,
            verbosity=self.verbosity
        )

        self.tabular_predictor_ = tabular_predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            auto_stack=self.auto_stack
        )
        
        return self

    def predict(self, X: XType) -> np.ndarray:
        check_is_fitted(self, "tabular_predictor_")
        X = self._prepare_X_to_predict(X)
        return self.tabular_predictor_.predict(X, as_pandas=False)

    def predict_proba(self, X:XType) -> np.ndarray:
        check_is_fitted(self, "tabular_predictor_")
        X = self._prepare_X_to_predict(X)
        return self.tabular_predictor_.predict_proba(X, as_pandas=False)

    @staticmethod
    def _prepare_X_to_predict(X: XType) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = [f"col_{i}" for i in range(X.shape[1])]
        return X



class AutoGluonSpec:
    classifier_class = AutoGluonClassifier
    early_stop_on_validation_set = False
    random_state_parameter = None
    n_threads_parameter = "num_cpus"
    device_parameter = None
    main_device = "cuda"
    supported_devices = ["cpu", "cuda"]
    default_preprocessing = "density_filter"
    default_params = {
        "eval_metric": "log_loss",
        "presets": "extreme_quality",
        "time_limit": 7200, # 2 hours
    }
    callbacks_on_params = None