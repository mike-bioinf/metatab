import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope
from metatab.hp_search.tabpfn_search_space import TABPFN_TUNE_SPACE



## FIXME: we cannot differentiate between same-named parameters for different estimators.
HPS_MIXED_TYPED = [
    # for random_forest and extra_trees
    "max_features",
    # this two tabpfn HP are listed here to avoid the FutureWarning raised by pandas concat:
    # """The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. 
    # In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. 
    # To retain the old behavior, exclude the relevant entries before the concat operation."""
    "inference_config__OUTLIER_REMOVAL_STD",
    "inference_config__SUBSAMPLE_SAMPLES"
]


def add_preprocessing_to_cls_search_space(cls_search_space: dict) -> dict:
    '''
    Adds the preprocessing option to the hyperopt classifier search space.
    Returns the modified search space.
    '''
    return {
        **cls_search_space, 
        "preprocessing": hp.choice("preprocessing", ["no", "base"]) ##REVIEW: add updated preprocessing options
    }


class TuningParams:
    '''
    Class that contains the configurations of parameters to tune, and the configuration 
    of parameters to set to fixed values (referred as fixed params) for all estimators. 
    Note that the fixed ones can be set to values that differ from the library defaults.
    '''

    ### RANDOM FOREST ------------------------------------------------------------------------------
    RF_FIXED_PARAMS = {
        "n_estimators": 1000
    }

    ## TODO:FUTURE: add criterion when you update the prior
    RF = {
        "max_features": hp.choice("max_features", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"]),
        "min_samples_split": hp.choice("min_samples_split", list(range(2, 16))),
        "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 4, 5]),
        "max_samples": hp.choice("max_samples", [0.7, 0.8, 0.9, 1.0]),
        "min_impurity_decrease": hp.choice("min_impurity_decrease", [0, hp.loguniform("mid_positive", np.log(1e-5), np.log(1e-3))])
    }

    ### EXTRA TREES -----------------------------------------------------------------------------------
    EXTRA_TREES_FIXED_PARAMS = {
        "n_estimators": 1000
    }

    EXTRA_TREES = {
        "max_features": hp.choice("max_features", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None, "sqrt", "log2"]),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "min_samples_split": hp.choice("min_samples_split", list(range(2, 16))),
        "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 4, 5]),
        "min_impurity_decrease": hp.choice("min_impurity_decrease", [0, hp.loguniform("mid_positive", np.log(1e-5), np.log(1e-3))])
    }

    ### XGBOOST ---------------------------------------------------------------------------------------
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
    XGB = {
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

    ### CATBOOST ----------------------------------------------------------------------------------------

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
    CATBOOST = {
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

    ### LIGHTGBM ----------------------------------------------------------------------------
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
    LGMB = {
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

    ### TABPFN --------------------------------------------------------------------------------
    # Here we use the search space defined in the "official extension" of tuned tabpfn with minor modifications. 
    # "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/hpo/search_space.py".

    TABPFN_FIXED_PARAMS = {
        "ignore_pretraining_limits": True
    }

    TABPFN = TABPFN_TUNE_SPACE

    ### REALMLP ---------------------------------------------------------------------------------
    # We use the autogluon/tabarena space with minor modifications.
    # "https://github.com/autogluon/tabarena/blob/main/tabarena/tabarena/models/realmlp/generate.py"

    REALMLP_FIXED_PARAMS = {
        # we double the default of 256 since we work with small datasets and so each epoch is made of few steps
        "n_epochs": 512, # increase in time
        "val_metric_name": "cross_entropy",
        # is suggested by author to set label smoothing to False when you are interested in AUC/log-loss
        "use_ls": False,
        "n_ens": 8 # increase in time and memory peak
    }

    REALMLP = {
        "batch_size": hp.choice("batch_size", ["auto", 256]),
        "hidden_sizes": hp.choice("hidden_sizes", ["rectangular"]),
        "n_hidden_layers": hp.choice("n_hidden_layers", [2, 3, 4]),
        "hidden_width": hp.choice("hidden_width", [256, 384, 512]), # increase in time and memory
        "tfms": hp.choice("tfms", [[], ["median_center", "robust_scale", "smooth_clip"]]), # none or default preprocessing
        "plr_sigma": hp.loguniform("plr_sigma", np.log(1e-2), np.log(50)),
        "plr_hidden_1": hp.choice("plr_hidden_1", [8, 16, 32]), # have a minor-moderate impact on time and memory peak
        "plr_hidden_2": hp.choice("plr_hidden_2", [4, 6, 8, 12]), # have a large impact on time and memory peak
        "plr_lr_factor": hp.loguniform("plr_lr_factor", np.log(5e-2), np.log(3e-1)),
        "p_drop": hp.uniform("p_drop", 0.0, 0.5),
        "scale_lr_factor": hp.loguniform("scale_lr_factor", np.log(2.0), np.log(10.0)),
        "first_layer_lr_factor": hp.loguniform("first_layer_lr_factor", np.log(0.3), np.log(1.5)),
        "lr": hp.loguniform("lr", np.log(2e-2), np.log(3e-1)),
        "wd": hp.loguniform("wd", np.log(1e-3), np.log(5e-2)),
        "use_early_stopping": hp.choice("use_early_stopping", [False, True]), # can help in reducing computational time
        "early_stopping_additive_patience": hp.choice("early_stopping_additive_patience", [60]) # we x3 the default of 20 to be less aggressive
    }

    #### TABM --------------------------------------------------------------------------------------------------------------------    
    # We use the autogluon/tabarena space with minor modifications.
    # "https://github.com/autogluon/tabarena/blob/main/tabarena/tabarena/models/tabm/generate.py"

    TABM_FIXED_PARAMS = {
        "val_metric_name": "cross_entropy",
        # we increase the patience since epochs with small data are made of few steps
        "patience": 128,
        "gradient_clipping_norm": 1,
        # in tabm paper it shown that using same or different batches lead to no differences in performance
        # however using the same batch uses less ram
        "share_training_batches": True,
        # mixed precision should speed-up training on GPU
        "allow_amp": True
    }

    TABM = {
        "arch_type": hp.choice("arch_type", ["tabm", "tabm-mini"]),
        "num_emb_type": hp.choice("num_emb_type", ["pwl"]),
        "num_emb_n_bins": hp.choice("num_emb_n_bins", list(range(8, 129, 2))),
        "d_embedding": hp.choice("d_embedding", [8, 12, 16, 20, 24]), # high increase in time and memory peak
        "batch_size": hp.choice("batch_size", ["auto", 256]),
        "lr": hp.loguniform("lr", np.log(1e-4), np.log(3e-3)),
        "weight_decay": hp.choice("weight_decay", [0.0, hp.loguniform("pos_weight_decay", np.log(1e-4), np.log(1e-1))]),
        "d_block": hp.choice("d_block", list(range(128, 769, 32))), # high increase in time and memory peak
        "n_blocks": hp.choice("n_blocks", [2, 3, 4, 5]), # high increase in time
        "dropout": hp.choice("dropout", [0.0, hp.uniform("pos_dropout", 0.0, 0.5)]),
        "tfms": hp.choice("tfms", [[], ["quantile_tabr"], ["median_center", "robust_scale", "smooth_clip"]]), # none, tabm default and realmlp default preprocessing
    }