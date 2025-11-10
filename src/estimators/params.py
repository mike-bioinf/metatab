import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope
from hp_search.tabpfn_search_space import TABPFN_TUNE_SPACE
from estimators.types import TUNABLE_ESTIMATOR_TYPE



DEFAULT_TUNE_CONFIGURATION = {
    "configuration": "default",
    "algo": "tpe",
    "n_iter": 100,
    "n_repeats": 1,
    "n_splits": 5
}


## TODO: we do not differentiate between same-named parameters for different estimators. This is "manually" guaranteed.
# List of HPs that assume mixed typed values.
HPS_MIXED_TYPED = [
    "max_features",
    # this two tabpfn HP are listed here to avoid the FutureWarning raised by pandas concat:
    # """The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. 
    # In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. 
    # To retain the old behavior, exclude the relevant entries before the concat operation."""
    "inference_config__OUTLIER_REMOVAL_STD",
    "inference_config__SUBSAMPLE_SAMPLES"
]




class TuningParams:
    '''
    Class that contains the configurations of parameters to tune, and the configuration 
    of parameters to set to fixed values (referred as fixed params) for all estimators. 
    Note that the fixed ones can be set to values that differ from the library defaults.
    '''

    ### RANDOM FOREST ------------------------------------------------------------------------------
    RANDOM_FOREST_FIXED_PARAMS = {
        "n_estimators": 1000
    }

    RF_C0 = {
        "max_features": hp.choice("max_features", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"]),
        "min_samples_split": hp.choice("min_samples_split", list(range(2, 16))),
        "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 4, 5]),
        "max_samples": hp.choice("max_samples", [0.7, 0.8, 0.9, 1.0]),
        "min_impurity_decrease": hp.choice("min_impurity_decrease", [0, hp.loguniform("mid_positive", np.log(1e-5), np.log(1e-3))])
    }


    ### XGBOOST ---------------------------------------------------------------------------------------
    # We explore different quantization-tree_growing_policy combinations.
    # We also consider more and less regularized configurations for most scenario.
    # With small sparse datasets we expect the quantization methods to converge especially 
    # with higher max_bin values. This is true especially for "hist" and "exact" variants. 
    # Therefore we mainly explore fixed quantization-tree_policies. 
    # We do not explore less regularized configuration for "approx" algo since it's slow.


    XGB_FIXED_PARAMS = {
        "n_estimators": 1000,
        "verbosity": 0
    }

    ES_XGB_FIXED_PARAMS = {
        "n_estimators": 10000,
        "eval_metric": "logloss_to_adjust",
        "verbosity": 0
    }

    # depthwise-exact-strong_regularized
    XGB_C0 = {
        "grow_policy": hp.choice("grow_policy", ["depthwise"]),
        "tree_method": hp.choice("tree_method", ["exact"]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bylevel": hp.choice("colsample_bylevel", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # lossguide-approx-strong_regularized
    XGB_C1 = {
        "grow_policy": hp.choice("grow_policy", ["lossguide"]),
        "tree_method": hp.choice("tree_method", ["approx"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 256]),
        "max_depth": hp.choice("max_depth", [0]), # no constrain
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(128), q=1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # lossguide-hist-strong_regularized
    XGB_C2 = {
        "grow_policy": hp.choice("grow_policy", ["lossguide"]),
        "tree_method": hp.choice("tree_method", ["hist"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 256]),
        "max_depth": hp.choice("max_depth", [0]), # no constrain
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(128), q=1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # depthwise-exact-weak_regularization
    XGB_C3 = {
        "grow_policy": hp.choice("grow_policy", ["depthwise"]),
        "tree_method": hp.choice("tree_method", ["exact"]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.choice("reg_lambda", [0, hp.loguniform("lambda_positive", np.log(0.001), np.log(5))]),
        "reg_alpha": hp.choice("reg_alpha", [0, hp.loguniform("alpha_positive", np.log(0.001), np.log(5))]),
        "gamma": hp.choice("gamma", [0, hp.loguniform("gamma_positive", np.log(0.001), np.log(5))]),
        "min_child_weight": hp.choice("min_child_weight", [0, hp.loguniform("mcw_positive", np.log(0.001), np.log(5))]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bylevel": hp.choice("colsample_bylevel", [0.6, 0.7, 0.8, 0.9, 1])
    }

    ## lossguide-hist-weak_regularization
    XGB_C4 = {
        "grow_policy": hp.choice("grow_policy", ["lossguide"]),
        "tree_method": hp.choice("tree_method", ["hist"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 256]),
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(128), q=1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.choice("reg_lambda", [0, hp.loguniform("lambda_positive", np.log(0.001), np.log(5))]),
        "reg_alpha": hp.choice("reg_alpha", [0, hp.loguniform("alpha_positive", np.log(0.001), np.log(5))]),
        "gamma": hp.choice("gamma", [0, hp.loguniform("gamma_positive", np.log(0.001), np.log(5))]),
        "min_child_weight": hp.choice("min_child_weight", [0, hp.loguniform("mcw_positive", np.log(0.001), np.log(5))]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bylevel": hp.choice("colsample_bylevel", [0.6, 0.7, 0.8, 0.9, 1])
    }



    ### CATBOOST ----------------------------------------------------------------------------------------
    # We explore the different combinations of split quality score metrics and tree growing policies.
    # Hovewer we do not explore the split metrics requiring GPU training, 
    # since it is non-deterministic (NetwonCosine and NewtonL2 metrics).
    # We keep the defaults when in comes to leaf estimation method and split finding algo.
    # We do not try the boosting type "Ordered" since is too slow.


    ## we list also the library defaults that we use just to be explicit
    CATBOOST_FIXED_PARAMS = {
        "n_estimators": 1000,  # default
        "leaf_estimation_method": "Newton", # default
        "feature_border_type": 'GreedyLogSum', # default
        "bootstrap_type": "Bayesian", # default
        "verbose": False,
        "allow_writing_files": False
    }

    ## we list also the library defaults that we use just to be explicit
    ES_CATBOOST_FIXED_PARAMS = {
        "n_estimators": 10000,
        "eval_metric": "logloss_to_adjust",
        "od_type":"Iter", # classical early stop on validation set
        "use_best_model": True, # select early stopped ensemble
        "leaf_estimation_method": "Newton",
        "feature_border_type": 'GreedyLogSum',
        "bootstrap_type": "Bayesian",
        "verbose": False,
        "allow_writing_files": False
    }

    # Cosine-Symmetrictree
    CATBOOST_C0 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["SymmetricTree"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }
    
    # Cosine-Depthwise
    CATBOOST_C1 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["Depthwise"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with Depthwise and Lossguide
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # Cosine-Lossguide
    CATBOOST_C2 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["Lossguide"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(128), q=1)),
        "max_depth": hp.choice("max_depth", [16]), # in catboost the depth must be always set (16 is the default with lossguide)
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # L2-Symmetrictree
    CATBOOST_C3 = {
        "score_function": hp.choice("score_function", ["L2"]),
        "grow_policy": hp.choice("grow_policy", ["SymmetricTree"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # L2-Depthwise
    CATBOOST_C4 = {
        "score_function": hp.choice("score_function", ["L2"]),
        "grow_policy": hp.choice("grow_policy", ["Depthwise"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # L2-Lossguide
    CATBOOST_C5 = {
        "score_function": hp.choice("score_function", ["L2"]),
        "grow_policy": hp.choice("grow_policy", ["Lossguide"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(128), q=1)),
        "max_depth": hp.choice("max_depth", [16]), # in catboost the depth must be always set (16 is the default with lossguide)
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # C0 with "Ordered" boosting type
    CATBOOST_C6 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["SymmetricTree"]),
        "boosting_type": hp.choice("boosting_type", ["Ordered"]),
        # we reduce max_bin vs other spaces to speed up
        "max_bin": hp.choice("max_bin", [5, 10, 15, 50]),
        "max_depth": hp.choice("max_depth", list(range(1, 9))),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }



    ### LIGHTGBM -----------------------------------------------------------
    # Lightgmb offer less variability in terms of algo variants in many/all
    # aspects of a GBDT framework. Therefore we use the only variant available 
    # with strong and weak regularization.
    # We do not try other weak learners (dart and rf) to be consistent with 
    # the other gbdts.
    
     
    # we list also the library defaults that we use just to be explicit
    LGBM_FIXED_PARAMS = {
        "n_estimators": 1000, # higher than default 100
        "boosting_type": "gbdt", # dart and rf are also possible (default)
        "max_depth": -1, # no control (default)
        "data_sample_strategy": "bagging", # more robust than goss (default)
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True
    }

    # we list also the library defaults that we use just to be explicit
    ES_LGBM_FIXED_PARAMS = {
        "n_estimators": 10000,
        "boosting_type": "gbdt", # dart and rf are also possible (default)
        "max_depth": -1, # no control (default)
        "data_sample_strategy": "bagging", # more robust than goss (default)
        "verbose": -1,
        "early_stopping_min_delta": 0, # to avoid premature stopping (default)
        "metric": "logloss_to_adjust",
        "deterministic": True,
        "force_col_wise": True
    }
    
    # strong-regularized configuration
    LGMB_C0 = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "num_leaves": scope.int(hp.qloguniform("num_leaves", np.log(2), np.log(128), 1)),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 255]),
        "min_data_in_bin": hp.choice("min_data_in_bin", list(range(1, 6))),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "min_split_gain": hp.loguniform("min_split_gain", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "min_child_samples": hp.choice("min_child_samples", list(range(1, 6))),
        "extra_trees": hp.choice("extra_trees", [False, True]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "subsample_freq": hp.choice("subsample_freq", [1]), # subsample every tree
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # weak-regularized configuration.
    # It is prone to error --> Check failed: (best_split_info.{left/right}_count) > (0),
    # with small datasets
    LGMB_C1 = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "num_leaves": scope.int(hp.qloguniform("num_leaves", np.log(2), np.log(128), 1)),
        "max_bin": hp.choice("max_bin", [5, 10, 20, 30, 50, 100, 150, 254]),
        "min_data_in_bin": hp.choice("min_data_in_bin", list(range(1, 6))),
        "reg_lambda": hp.choice("reg_lambda", [0, hp.loguniform("lambda_positive", np.log(0.001), np.log(5))]),
        "reg_alpha": hp.choice("reg_alpha", [0, hp.loguniform("alpha_positive", np.log(0.001), np.log(5))]),
        "min_split_gain": hp.choice("min_split_gain", [0, hp.loguniform("min_split_gain_positive", np.log(0.001), np.log(5))]),
        "min_child_weight": hp.choice("min_child_weight", [0, hp.loguniform("min_child_weight_positive", np.log(0.001), np.log(5))]),
        "min_child_samples": hp.choice("min_child_samples", list(range(1, 6))),
        "extra_trees": hp.choice("extra_trees", [False, True]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "subsample_freq": hp.choice("subsample_freq", [1, 2, 3, 4, 5]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1]),
    }


    ### TABPFN --------------------------------------------------------
    # Here we use the search space defined in the "official extension" of tuned tabpfn with minor modifications. 
    # "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/hpo/search_space.py".

    TABPFN_FIXED_PARAMS = {
        "ignore_pretraining_limits": True
    }

    TABPFN_C0 = TABPFN_TUNE_SPACE





# Dict of default tuning spaces for each tunable estimator.
# The default spaces are identified based on our paper preanalysis
DEFAULT_ESTIMATORS_TUNE_SPACES = {
    "random_forest": ("c0", TuningParams.RF_C0),
    "xgb": ("c0", TuningParams.XGB_C0), 
    "es_xgb": ("c0", TuningParams.XGB_C0), 
    "catboost": ("c0", TuningParams.CATBOOST_C0), 
    "es_catboost": ("c0", TuningParams.CATBOOST_C0),
    "lgbm": ("c0", TuningParams.LGMB_C0), 
    "es_lgbm": ("c0", TuningParams.LGMB_C0),
    "tabpfn": ("c0", TuningParams.TABPFN_C0)
}



def pick_estimator_space(space: str, estimator: TUNABLE_ESTIMATOR_TYPE) -> dict:
     match (estimator, space):
        case ("random_forest", "c0"):
            return TuningParams.RF_C0
        
        case ("xgb" | "es_xgb", "c0"):
            return TuningParams.XGB_C0
        case ("xgb" | "es_xgb", "c1"):
            return TuningParams.XGB_C1
        case ("xgb" | "es_xgb", "c2"):
            return TuningParams.XGB_C2
        case ("xgb" | "es_xgb", "c3"):
            return TuningParams.XGB_C3
        case ("xgb" | "es_xgb", "c4"):
            return TuningParams.XGB_C4
        
        case ("catboost" | "es_catboost", "c0"):
            return TuningParams.CATBOOST_C0
        case ("catboost" | "es_catboost", "c1"):
            return TuningParams.CATBOOST_C1
        case ("catboost" | "es_catboost", "c2"):
            return TuningParams.CATBOOST_C2
        case ("catboost" | "es_catboost", "c3"):
            return TuningParams.CATBOOST_C3
        case ("catboost" | "es_catboost", "c4"):
            return TuningParams.CATBOOST_C4
        case ("catboost" | "es_catboost", "c5"):
            return TuningParams.CATBOOST_C5
        case ("catboost" | "es_catboost", "c6"):
            return TuningParams.CATBOOST_C6

        case ("lgbm" | "es_lgbm", "c0"):
            return TuningParams.LGMB_C0
        case ("lgbm" | "es_lgbm", "c1"):
            return TuningParams.LGMB_C1

        case ("tabpfn", "c0"):
            return TuningParams.TABPFN_C0
        
        case (_, "default"):
            return DEFAULT_ESTIMATORS_TUNE_SPACES[estimator][1]
            
        case _:
            raise ValueError(
                f"Unsupported configuration '{space}' for '{estimator}' estimator."
            )





class DefaultParams:
    '''
    Class of the default library level configurations.
    These configurations set only "system-level" parameters.
    We allow for some execptions when the default values lead to 
    errors/wrong behaviours, or when no default exists, like for 
    the parameters controlling the early stop procedure for some estimators.
    '''
    
    RANDOM_FOREST_DEFAULT_PARAMS = {}

    XGB_DEFAULT_PARAMS = {
        "verbosity": 0
    }
    
    # we raise the default number of trees since with early stop 
    # the "default behaviour" is to indirectly control this parameter via training
    ES_XGB_DEFAULT_PARAMS = {
        "n_estimators": 10000,
        "eval_metric": "logloss_to_adjust",
        "verbosity": 0
    }

    CATBOOST_DEFAULT_PARAMS = {
        "verbose": False,
        "allow_writing_files": False
    }

    # we raise the default number of trees since with early stop 
    # the "default behaviour" is to indirectly control this parameter via training
    ES_CATBOOST_DEFAULT_PARAMS = {
        "n_estimators": 10000,
        "od_type": "Iter", 
        "eval_metric": "logloss_to_adjust",
        "use_best_model": True,
        "verbose": False,
        "allow_writing_files": False
    }

    # we set min_child_samples to 1 since the default 20 
    # does not permit the tree building on small datasets.
    LGBM_DEFAULT_PARAMS = {
        "min_child_samples": 1,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True
    }

    # we raise the default number of trees since with early stop 
    # the "default behaviour" is to indirectly control this parameter via training.
    ES_LGBM_DEFAULT_PARAMS = {
        "n_estimators": 10000,
        "metric": "logloss_to_adjust",
        "min_child_samples": 1,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True
    }
    
    TABPFN_DEFAULT_PARAMS = {
        "ignore_pretraining_limits": True,
        # suppressing categorical transformation 
        # that leads to testing data loss with small sparse data
        "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0}
    }

    AUTOTABPFN_DEFAULT_PARAMS = {
        # TODO: keep max_time fixed at 3 hour or adapt it depending on the scenario in some way?
        # max_time of 3 hours set based on 7 days job time wall and classic 50 iteration 
        # done in a resampling strategy. However sometimes we do not use 50 repeats.
        # In addition we use this value also in the fit program where we fit one time.
        "max_time": 10800,   # 3 hours
        "eval_metric": "log_loss",
        "presets": "best_quality",
        "phe_init_args": {"verbosity": 0}, # dict passed to autogluon TabularPredictor
        "n_ensemble_models": 20,
        "n_estimators": 8,
        "ignore_pretraining_limits": True
    }

    AESFINETUNEDTABPFN_DEFAULT_PARAMS = {
        "finetune_setup": {"max_steps": 10000}, # high value to stop on other conditions
        "tabpfn_classifier_params": {
            "ignore_pretraining_limits": True,
            "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0} 
        },
        # we finetune the post-trained model (realtabpfn) since is the one used for other tabpfn-based estimators
        "model_path": "default", 
        "learning_rate": 1e-5, # default
        "batch_size": 1, # default (currently enforced)
        "n_accumulation_steps": 10,  # more than default (1) since tabpfn is a meta-estimator
        "log": False
    }