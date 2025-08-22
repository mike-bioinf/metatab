import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope



DEFAULT_TUNE_CONFIGURATION = {
    "configuration": "c0",
    "algo": "tpe",
    "n_iter": 100,
    "n_repeats": 1,
    "n_splits": 5
}



class TuningParams:
    '''
    Class that contains the configurations of parameters to tune,
    and the configuration of parameters to set to fixed values,
    referred as fixed params, for all estimators. 
    Note that the last ones can be set to values different from the library defaults.
    '''

    ### RANDOM FOREST --------------------------------------------------------
    RANDOM_FOREST_FIXED_PARAMS = {
        "n_estimators": 1000
    }

    RF_C0 = {
        "max_features": hp.choice("max_features", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"]),
        "min_samples_split": hp.randint("min_samples_split", 2, 21),
        "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 4, 5]),
        "max_samples": hp.choice("max_samples", [0.6, 0.7, 0.8, 0.9, 1.0])
    }

    # extended version of C0
    RF_C1 = {
        "max_features": hp.choice("max_features", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"]),
        "min_samples_split": hp.randint("min_samples_split", 2, 21),
        "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 4, 5]),
        "max_samples": hp.choice("max_samples", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "bootstrap": hp.choice("bootstrap", [False, True]),
        "min_inpurity_decrease": hp.choice("min_inpurity_decrease", [0, hp.loguniform("mid_positive", np.log(1e-5), np.log(1e-3))])
    }


    ### XGBOOST ---------------------------------------------------------------------------------------
    # We explore different quantization-tree growing policy variants combinations.
    # We also consider more and less regularized configurations for most scenario.
    # With small sparse datasets the quantization methods converge, even though we 
    # can set a lower number of bins. This is True especially for "hist" and "exact" variants. 
    # Therefore we mainly explore the quantization variants in combination with different growing policy. 
    # We do not explore less regularized configuration for "approx" algo since it's slow.

    XGB_FIXED_PARAMS = {
        "n_estimators": 700,
        "verbosity": 0
    }

    ES_XGB_FIXED_PARAMS = {
        "n_estimators": 10000,
        "eval_metric": "logloss_to_adjust",
        "early_stopping_rounds": 50,
        "verbose_eval": False,
        "verbosity": 0
    }

    # depthwise-exact-strong_regularized
    XGB_C0 = {
        "grow_policy": hp.choice("grow_policy", ["depthwise"]),
        "tree_method": hp.choice("tree_method", ["exact"]),
        "max_depth": hp.randint("max_depth", 2, 9),
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
        # not sure about max_bin effects, so choice between the default 256 and a list of smaller values
        "max_bin": hp.choice("max_bin", [256, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
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
        # not sure about max_bin effects, so choice between the default 256 and a list of smaller values
        "max_bin": hp.choice("max_bin", [256, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
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
        "max_depth": hp.randint("max_depth", 2, 9),
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
        # not sure about max_bin effects, so choice between the default 256 and a list of smaller values
        "max_bin": hp.choice("max_bin", [256, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "max_depth": hp.randint("max_depth", 2, 9),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.choice("reg_lambda", [0, hp.loguniform("lambda_positive", np.log(0.001), np.log(5))]),
        "reg_alpha": hp.choice("reg_alpha", [0, hp.loguniform("alpha_positive", np.log(0.001), np.log(5))]),
        "gamma": hp.choice("gamma", [0, hp.loguniform("gamma_positive", np.log(0.001), np.log(5))]),
        "min_child_weight": hp.choice("min_child_weight", [0, hp.loguniform("mcw_positive", np.log(0.001), np.log(5))]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bylevel": hp.choice("colsample_bylevel", [0.6, 0.7, 0.8, 0.9, 1])
    }



    ### CATBOOST ----------------------------------------------------------------------------------------
    # We explore different variants in terms of split quality score metrics, 
    # tree and related boosting modes.
    # We do not explore the split metrics requiring GPU training 
    # since it is non-deterministic (NetwonCosine and NewtonL2 metrics).
    # We keep the defaults when in comes to leaf estimation method and split finding algos.
    # We tune also the max_bin parameter (quantization aspect) setting a choice between 
    # the default and a list of lower values since the effect of selecting smaller values 
    # is not clear at priori.


    ## we list also the library defaults that we use just to be explicit
    CATBOOST_FIXED_PARAMS = {
        "n_estimators": 1000,  ## more than xgboost since less prone to overfit (default)
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
        "early_stopping_rounds": 70,
        "od_type":"Iter", # classical early stop on validation set
        "use_best_model": True, # select early stopped ensemble
        "leaf_estimation_method": "Newton",
        "feature_border_type": 'GreedyLogSum',
        "bootstrap_type": "Bayesian",
        "verbose": False,
        "allow_writing_files": False
    }

    # Cosine-Symmetrictree-Ordered
    CATBOOST_C0 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["SymmetricTree"]),
        "boosting_type": hp.choice("boosting_type", ["Ordered"]),
        "max_bin": hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }
    
    # Cosine-Depthwise-Plain
    CATBOOST_C1 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["Depthwise"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]),  # Ordered is not possible with depthwise and lossguide
        "max_bin": hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with Depthwise and Lossguide
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # Cosine-Lossguide-Plain
    CATBOOST_C2 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["Lossguide"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]), # Ordered is not possible with depthwise and lossguide
        "max_bin": hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(512), q=1)),
        "max_depth": hp.choice("max_depth", [16, 20, 30, 50, 100]), # in catboost is not possible to not set a depth (16 is default with lossguide)
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # L2-Symmetrictree-Ordered
    CATBOOST_C3 = {
        "score_function": hp.choice("score_function", ["L2"]),
        "grow_policy": hp.choice("grow_policy", ["SymmetricTree"]),
        "boosting_type": hp.choice("boosting_type", ["Ordered"]),
        "max_bin": hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # L2-Depthwise-Plain
    CATBOOST_C4 = {
        "score_function": hp.choice("score_function", ["L2"]),
        "grow_policy": hp.choice("grow_policy", ["Depthwise"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]), # Ordered is not possible with depthwise and lossguide
        "max_bin": hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": scope.int(hp.qloguniform("lei", np.log(1), np.log(10), q=1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # L2-Lossguide-Plain
    CATBOOST_C5 = {
        "score_function": hp.choice("score_function", ["L2"]),
        "grow_policy": hp.choice("grow_policy", ["Lossguide"]),
        "boosting_type": hp.choice("boosting_type", ["Plain"]), # Ordered is not possible with depthwise and lossguide
        "max_bin": hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "max_leaves": scope.int(hp.qloguniform("max_leaves", np.log(2), np.log(512), q=1)),
        "max_depth": hp.choice("max_depth", [16, 20, 30, 50, 100]), # in catboost is not possible to not set a depth (16 is default with lossguide)
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
    # Similar to catboost and xgboost we tune the max_bin using a choice between
    # default and a list of smaller values.

    # we list also the library defaults that we use just to be explicit
    LGBM_FIXED_PARAMS = {
        "n_estimators": 700, # higher than default 100
        "boosting_type": "gbdt", # dart and rf are also possible (default)
        "max_depth": -1, # no control (default)
        "data_sample_strategy": "bagging", # more robust than goss (default)
        "verbose": -1
    }

    # we list also the library defaults that we use just to be explicit
    ES_LGBM_FIXED_PARAMS = {
        "n_estimators": 10000,
        "boosting_type": "gbdt", # dart and rf are also possible (default)
        "max_depth": -1, # no control (default)
        "data_sample_strategy": "bagging", # more robust than goss (default)
        "verbose": -1,
        "early_stopping_rounds": 50, # set to 50 as for xgboost
        "early_stopping_min_delta": 0, # to avoid premature stopping (default)
        "metric": "logloss_to_adjust"
    }
    
    # strong-regularized configuration
    LGMB_C0 = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "num_leaves": scope.int(hp.qloguniform("num_leaves", np.log(2), np.log(512), 1)),
        "max_bin": hp.choice("max_bin", [255, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "min_data_in_bin": hp.choice("min_data_in_bin", list(range(1, 11))),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "min_split_gain": hp.loguniform("min_split_gain", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "min_child_samples": hp.randint("min_child_samples", 1, 5),
        "extra_trees": hp.choice("extra_trees", [False, True]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "subsample_freq": hp.choice("subsample_freq", [1]), # subsample every tree
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }

    # weak-regularized configuration
    LGMB_C1 = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "num_leaves": scope.int(hp.qloguniform("num_leaves", np.log(2), np.log(512), 1)),
        "max_bin":  hp.choice("max_bin", [254, hp.choice("max_bin_positive", list(range(20, 250, 20)))]),
        "min_data_in_bin": hp.choice("min_data_in_bin", list(range(1, 11))),
        "reg_lambda": hp.choice("reg_lambda", [0, hp.loguniform("lambda_positive", np.log(0.001), np.log(5))]),
        "reg_alpha": hp.choice("reg_alpha", [0, hp.loguniform("alpha_positive", np.log(0.001), np.log(5))]),
        "min_split_gain": hp.choice("min_split_gain", [0, hp.loguniform("min_split_gain_positive", np.log(0.001), np.log(5))]),
        "min_child_weight": hp.choice("min_child_weight", [0, hp.loguniform("min_child_weight_positive", np.log(0.001), np.log(5))]),
        "min_child_samples": hp.randint("min_child_samples", 1, 5),
        "extra_trees": hp.choice("extra_trees", [False, True]),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "subsample_freq": hp.choice("subsample_freq", [1, 2, 3, 4, 5]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1]),
    }





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
        "verbose_eval": False,
        "eval_metric": "logloss_to_adjust",
        "early_stopping_rounds": 50,
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
        "early_stopping_rounds": 70,
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
        "verbose": -1
    }

    # we raise the default number of trees since with early stop 
    # the "default behaviour" is to indirectly control this parameter via training.
    ES_LGBM_DEFAULT_PARAMS = {
        "n_estimators": 10000,
        "early_stopping_rounds": 50,
        "metric": "logloss_to_adjust",
        "min_child_samples": 1,
        "verbose": -1
    }
    
    TABPFN_DEFAULT_PARAMS = {
        "ignore_pretraining_limits": True,
        # suppressing categorical transformation 
        # that leads to testing data loss with sparse data
        "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0}
    }