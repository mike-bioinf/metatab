import numpy as np
from hyperopt import hp



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
    referred as fixed params,for all estimators. 
    Note that the last ones are/can be different from the library defaults.
    '''

    ### XGBOOST ---------------------------------------------------------------------------------------
    # We explore different quantization methods with different tree growing policy.
    # We also consider more and less regularized configurations for most cases.
    # With small sparse datasets the quantization methods converge. This is True 
    # especially for "hist" and "exact" variants. Therefore we mainly explore them with
    # different growing policy. We do not explore less regularized configuration
    # for "approx" algo since it's slow.

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

    XGB_C1 = {
        "grow_policy": hp.choice("grow_policy", ["lossguide"]),
        "tree_method": hp.choice("tree_method", ["approx"]),
        "max_depth": hp.choice("max_depth", [0]),
        "max_leaves": hp.qloguniform("max_leaves", np.log(4), np.log(128), q=1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }

    ## same as C1 expect for hist quantization instaed of approx
    XGB_C2 = {
        "grow_policy": hp.choice("grow_policy", ["lossguide"]),
        "tree_method": hp.choice("tree_method", ["hist"]),
        "max_depth": hp.choice("max_depth", [0]),
        "max_leaves": hp.qloguniform("max_leaves", np.log(4), np.log(128), q=1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.001), np.log(5)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(5)),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(5)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(5)),
        "subsample": hp.choice("subsample", [0.8, 0.9, 1]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1])
    }

    ## depthwise-exact with lower regularization
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

    ## lossguide-hist with lower regularization
    XGB_C4 = {
        "grow_policy": hp.choice("grow_policy", ["lossguide"]),
        "tree_method": hp.choice("tree_method", ["hist"]),
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


    ## we list also the library defaults that we use just to be explicit
    CATBOOST_FIXED_PARAMS = {
        "n_estimators": 1000,  ## more than xgboost since less prone to overfit
        "leaf_estimation_method": "Newton",
        "feature_border_type": 'GreedyLogSum',
        "border_count": 254,
        "bootstrap_type": "Bayesian",
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
        "border_count": 254,
        "bootstrap_type": "Bayesian",
        "verbose": False,
        "allow_writing_files": False
    }

    # Cosine-Symmetrictree-Ordered
    CATBOOST_C0 = {
        "score_function": hp.choice("score_function", ["Cosine"]),
        "grow_policy": hp.choice("grow_policy", ["SymmetricTree"]),
        "boosting_type": hp.choice("boosting_type", ["Ordered"]),
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": hp.qloguniform("lei", np.log(1), np.log(10), q=1),
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
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with Depthwise and Lossguide
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": hp.qloguniform("lei", np.log(1), np.log(10), q=1),
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
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "max_leaves": hp.randint("max_leaves", 4, 1025),  # we control the number of leaves and not the depth with lossguide
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": hp.qloguniform("lei", np.log(1), np.log(10), q=1),
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
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": hp.qloguniform("lei", np.log(1), np.log(10), q=1),
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
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "max_depth": hp.randint("max_depth", 2, 11),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": hp.qloguniform("lei", np.log(1), np.log(10), q=1),
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
        "min_data_in_leaf": hp.choice("min_data_in_leaf", [1, 2, 3, 4, 5]), # work only with depthwise and lossguide
        "max_leaves": hp.randint("max_leaves", 4, 1025),  # we control the number of leaves and not the depth with lossguide
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
        "leaf_estimation_iterations": hp.qloguniform("lei", np.log(1), np.log(10), q=1),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-4), np.log(5)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
        "random_strength": hp.quniform("random_strength", 1, 11, 1),
        "rsm": hp.choice("rsm", [0.6, 0.7, 0.8, 0.9, 1])
    }


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





class DefaultParams:
    '''
    Class of the default library level configurations.
    These configurations set only "system-level" parameters.
    We allow for some execptions when the default values lead to 
    errors/wrong behaviours, or when no default exists, like for 
    the parameters controlling the early stop procedure for some estimators.
    '''

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

    RANDOM_FOREST_DEFAULT_PARAMS = {}

    TABPFN_DEFAULT_PARAMS = {
        "ignore_pretraining_limits": True,
        # suppressing categorical transformation 
        # that leads to testing data loss with sparse data
        "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0}
    }