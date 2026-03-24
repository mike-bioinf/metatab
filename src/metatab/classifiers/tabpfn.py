"""
In this module we define a predifined search space for the tabpfn classifiers.
This is mostly taken from the one proposed in the "official extension" available at:
"https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/hpo".
We also use tabpfn internal capabilities to locally download and store tabpfn classifier checkpoints.
"""
import sys
import optuna
from tabpfn import TabPFNClassifier
from tabpfn.model_loading import _user_cache_dir, download_model
 


TABPFN_CHECKPOINTS = [
    "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt", # is the new default
    "tabpfn-v2-classifier.ckpt", # old default
    "tabpfn-v2-classifier-od3j1g5m.ckpt",
    "tabpfn-v2-classifier-gn2p4bpt.ckpt",
    "tabpfn-v2-classifier-znskzxi4.ckpt",
    "tabpfn-v2-classifier-llderlii.ckpt",
    "tabpfn-v2-classifier-vutqq28w.ckpt"
]


def _download_and_return_tabpfn_checkpoints(ckpts: str | list[str]) -> list[str]:
    '''
    Download the input tabpfn checkpoints in the user cache directory
    identified by the PriorLabs `_user_cache_dir` utility.
    Returns the list of checkpoint absolute paths as strings.   
    '''
    ckpts = ckpts if isinstance(ckpts, list) else [ckpts]
    # get the absolute path where models live or will be downloaded
    local_dir = _user_cache_dir(sys.platform, appname="tabpfn").resolve()
    # we assure that the folder exists
    local_dir.mkdir(parents=True, exist_ok=True)

    ckpts_paths = []

    for ckpt in ckpts:
        ckpt_file = local_dir / ckpt
        if not ckpt_file.exists():
            download_result = download_model(
                to=ckpt_file, # here we must include the ckpt filename
                version="v2",
                which="classifier",
                model_name=ckpt
            )    
            if download_result != "ok":
                raise RuntimeError(f"Download failed for {ckpt}: {download_result}")
        ckpts_paths.append(str(ckpt_file))

    return ckpts_paths


def enumerate_preprocess_transforms() -> list[list[dict]]:
    '''
    Generate a list of sublists of dicts of preprocessing instructions.
    Taken from "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/hpo/search_space.py",
    with minor modifications.
    '''
    transforms = []

    names_list = [
        ["safepower"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        ["quantile_uni"],
        ["none"],
        ["robust"],
        ["safepower", "quantile_uni"],
        ["none", "safepower"],
    ]

    for names in names_list:
        for append_original in [True, False]:
            for subsample_features in [-1, 0.99, 0.95, 0.9]:
                for global_transformer_name in [None, "svd"]:
                    transforms += [
                        [
                            {
                                "append_original": append_original,
                                # categorical features are treated as numeric,
                                # this a safe fallback in case tabpfn still treats some 
                                # features as categoricals even though we enforce only continuos features. 
                                "categorical_name": "numeric",
                                "global_transformer_name": global_transformer_name,
                                # Use "name" parameter as expected by TabPFN PreprocessorConfig
                                "name": name,
                                "subsample_features": subsample_features,
                            }
                            for name in names
                        ]
                    ]
    return transforms


def _tabpfn_sampler_function(trial: optuna.Trial) -> dict:
    point = {
        "model_path": trial.suggest_categorical("model_path", TABPFN_CHECKPOINTS),
        "n_estimators": trial.suggest_categorical("n_estimators", [8, 12, 16]),
        "average_before_softmax": trial.suggest_categorical("average_before_softmax", [True, False]),
        "softmax_temperature": trial.suggest_categorical("softmax_temperature", [0.75, 0.8, 0.9, 0.95, 1.0]),
        # inference config parameters
        "inference_config__FINGERPRINT_FEATURE": trial.suggest_categorical("inference_config__FINGERPRINT_FEATURE", [True, False]),
        "inference_config__PREPROCESS_TRANSFORMS": trial.suggest_categorical(
            "inference_config__PREPROCESS_TRANSFORMS",
            enumerate_preprocess_transforms()
        ),
        "inference_config__OUTLIER_REMOVAL_STD": trial.suggest_categorical(
            "inference_config__OUTLIER_REMOVAL_STD", [None, 7.0, 9.0, 12.0]
        ),
        "inference_config__SUBSAMPLE_SAMPLES": trial.suggest_categorical(
            "inference_config__SUBSAMPLE_SAMPLES", [0.99, None]
        )
    }
    return point


class TabPFNSpec:
    type_classifier = "tabpfn"
    classifier_class = TabPFNClassifier
    early_stop_on_validation_set = False
    random_state_parameter = "random_state"
    n_threads_parameter = "n_jobs"
    device_parameter = "device"
    main_device = "cuda"
    supported_devices = ["cpu", "cuda"]
    default_preprocessing = "density_filter"
    default_params = {
        "ignore_pretraining_limits": True,
        # suppressing categorical transformation that leads to testing data loss with small sparse data
        "inference_config": {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0},
    }
    fixed_params = {
        "ignore_pretraining_limits": True,
        "inference_config": {
            # suppressing categorical transformation that leads to testing data loss with small sparse data
            "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0,
            # avoid polynomial feature computation errors
            "POLYNOMIAL_FEATURES": "no"
        }
    }
    callbacks_on_params = None
    hps_sampler_function = _tabpfn_sampler_function
    initialize_search_function = lambda: _download_and_return_tabpfn_checkpoints(TABPFN_CHECKPOINTS)
    #refactor: check if these needed to avoid pandas warning
    params_as_object_columns_in_df_search = [
        "inference_config__OUTLIER_REMOVAL_STD",
        "inference_config__SUBSAMPLE_SAMPLES"
    ]