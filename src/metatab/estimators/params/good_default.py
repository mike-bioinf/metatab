class GoodDefaultParams:
    '''
    Class of the default configurations with changes to the library default based on personal expertise.
    '''
    RANDOM_FOREST_DEFAULT_PARAMS = {
        "n_estimators": 1000
    }

    EXTRA_TREES_DEFAULT_PARAMS = {
        "n_estimators": 1000
    }

    XGB_DEFAULT_PARAMS = {
        "n_estimators": 1000,
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

    LGBM_DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "min_child_samples": 3,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True
    }

    # we raise the default number of trees since with early stop 
    # the "default behaviour" is to indirectly control this parameter via training.
    ES_LGBM_DEFAULT_PARAMS = {
        "n_estimators": 10000,
        "metric": "logloss_to_adjust",
        "min_child_samples": 3,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True
    }
    
    TABPFN_DEFAULT_PARAMS = {
        "n_estimators": 16,
        "ignore_pretraining_limits": True,
        # suppressing categorical transformation 
        # that leads to testing data loss with small sparse data
        "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0}
    }

    REALMLP_DEFAULT_PARAMS = {
        "n_epochs": 512,
        "n_ens": 8,
        "val_metric_name": "cross_entropy",
        "n_hidden_layers": 4,
        "hidden_width": 512,
        "plr_hidden_1": 32,
        "plr_hidden_2": 12,
        "plr_lr_factor": 0.001,
        # is suggested by author to set label smooting to False when you are interested in AUC/log-loss
        "use_ls": False
    }

    TABM_DEFAULT_PARAMS = {
        "arch_type": "tabm",
        "num_emb_type": "pwl",
        "d_embedding": 16,
        "val_metric_name": "cross_entropy",
        # we try here the quantile transformation
        "tfms": ["quantile_tabr"],
        # we increase the patience since epochs with small data are made of few steps
        "patience": 128,
        "n_blocks": 5,
        "d_block": 512,
        # we differ from pytabkit using gradient clipping
        "gradient_clipping_norm": 1,
        # in tabm paper it shown that using same or different batches lead to no differences in performance
        # however using the same batch uses less ram
        "share_training_batches": True,
        # mixed precision should speed-up training on GPU
        "allow_amp": True
    }



GOOD_DEFAULTS_MAP = {
    "random_forest": GoodDefaultParams.RANDOM_FOREST_DEFAULT_PARAMS,
    "extra_trees": GoodDefaultParams.EXTRA_TREES_DEFAULT_PARAMS,
    "xgb": GoodDefaultParams.XGB_DEFAULT_PARAMS,
    "es_xgb": GoodDefaultParams.ES_XGB_DEFAULT_PARAMS,
    "catboost": GoodDefaultParams.CATBOOST_DEFAULT_PARAMS,
    "es_catboost": GoodDefaultParams.ES_CATBOOST_DEFAULT_PARAMS,
    "lgbm": GoodDefaultParams.LGBM_DEFAULT_PARAMS,
    "es_lgbm": GoodDefaultParams.ES_LGBM_DEFAULT_PARAMS,
    "tabpfn": GoodDefaultParams.TABPFN_DEFAULT_PARAMS,
    "realmlp": GoodDefaultParams.REALMLP_DEFAULT_PARAMS,
    "tabm": GoodDefaultParams.TABM_DEFAULT_PARAMS,
}