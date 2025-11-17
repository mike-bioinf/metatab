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