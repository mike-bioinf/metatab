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