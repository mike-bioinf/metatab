import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted



class SurrogateRandomForestRegressor(RegressorMixin, BaseEstimator):
    '''
    Class that implements a regression random forest that predicts values 
    and their uncertainty as the standard deviation of the forest trees predictions.

    The parameters are the ones accepted by "RandomForestRegressor" sklearn class, 
    with the same default values except for:
    - n_estimators: from 100 to 1000
    - min_samples_split: from 2 to 20
    - min_samples_leaf: from 1 to 5

    The last 2 changes help in avoid overfitting, which in this case should mean
    predicting performance scores based on a small number of meta-observations.

    Attributes:
    ------------------
    forest_: Fitted RandomForestRegressor instance.
    '''
    def __init__(
        self,
        *,
        n_estimators=1000,
        criterion='squared_error', 
        max_depth=None, 
        min_samples_split=20, 
        min_samples_leaf=5, 
        min_weight_fraction_leaf=0.0, 
        max_features=1.0, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        ccp_alpha=0.0, 
        max_samples=None, 
        monotonic_cst=None
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.monotonic_cst = monotonic_cst


    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray) -> "SurrogateRandomForestRegressor":
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )
        self.forest_ = forest.fit(X, y)
        return self


    def predict(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Predicts the values and their uncertainty.
        Returns a binary tuple of numpy arrays.
        '''
        check_is_fitted(self, "forest_")
        # we convert X to numpy since the trees of the forest are internally trained 
        # on numpy arrays and therefore they raise the warning:
        # "X has feature names, but DecisionTreeRegressor was fitted without feature names" 
        # when used for inference on dataframe.
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # tree_preds has shape (tree, pred)
        tree_preds = np.array([tree.predict(X) for tree in self.forest_.estimators_])
        # we compute the unbiased standard deviation with ddof = 1
        uncertainties = tree_preds.std(ddof=1, axis=0)
        forest_preds = tree_preds.mean(axis=0)
        return forest_preds, uncertainties