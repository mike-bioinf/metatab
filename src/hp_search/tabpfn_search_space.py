"""Search spaces for hyperparameter optimization of TabPFN classifiers

In this module we define a predifined search space for the tabpfn classifiers.
This is mostly taken from the one proposed in the "official extension" available at:
"https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/hpo".
We also use tabpfn internal capabilities to locally download and store tabpfn classifier checkpoints.
"""

import sys
from hyperopt import hp
from tabpfn.model_loading import _user_cache_dir, download_model



def enumerate_preprocess_transforms() -> list[dict]:
    '''
    Generate a list of dicts of preprocessing instructions.
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
                                    # Use "name" parameter as expected by TabPFN PreprocessorConfig
                                    "name": name,
                                    "global_transformer_name": global_transformer_name,
                                    "subsample_features": subsample_features,
                                    # categorical features are treated as numeric,
                                    # this a safe fallback in case tabpfn still treats some 
                                    # features as categoricals even though we enforce only continuos features. 
                                    "categorical_name": "numeric",
                                    "append_original": append_original
                                }
                                for name in names
                            ]
                        ]
    return transforms



def return_clf_paths_list() -> list:
    '''
    Ensure all TabPFN v2 classifier checkpoints exist in the cache directory.
    Downloads any missing models using PriorLabs functions.
    Returns the list of absolute paths to classifier checkpoints as strings.
    '''
    # return the absolute path where models live or will be downloaded
    local_dir = _user_cache_dir(sys.platform, appname="tabpfn").resolve()
    # we assure that the folder exists
    local_dir.mkdir(parents=True, exist_ok=True)

    ckpts = [
        "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt", # is the new default
        "tabpfn-v2-classifier.ckpt", # old default
        "tabpfn-v2-classifier-od3j1g5m.ckpt",
        "tabpfn-v2-classifier-gn2p4bpt.ckpt",
        "tabpfn-v2-classifier-znskzxi4.ckpt",
        "tabpfn-v2-classifier-llderlii.ckpt",
        "tabpfn-v2-classifier-vutqq28w.ckpt",
    ]
    
    model_paths = []

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
        model_paths.append(str(ckpt_file))

    return model_paths



TABPFN_TUNE_SPACE = {
    ## model parameters
    "model_path": hp.choice("model_path", return_clf_paths_list()),
    "n_estimators": hp.choice("n_estimators", [8, 12, 16]),
    "average_before_softmax": hp.choice("average_before_softmax", [True, False]),
    "softmax_temperature": hp.choice("softmax_temperature", [0.75, 0.8, 0.9, 0.95, 1.0]),
    ## inference config parameters
    "inference_config__FINGERPRINT_FEATURE": hp.choice("FINGERPRINT_FEATURE", [True, False]),
    "inference_config__PREPROCESS_TRANSFORMS": hp.choice("PREPROCESS_TRANSFORMS", enumerate_preprocess_transforms()),
    # only use "no" to avoid polynomial feature computation errors
    "inference_config__POLYNOMIAL_FEATURES": hp.choice("POLYNOMIAL_FEATURES", ["no"]),  
    "inference_config__OUTLIER_REMOVAL_STD": hp.choice("OUTLIER_REMOVAL_STD", [None, 7.0, 9.0, 12.0]),
    "inference_config__SUBSAMPLE_SAMPLES": hp.choice("SUBSAMPLE_SAMPLES", [0.99, None]),
    # we suppress categorical transformation since causes test data corruption with small sparse dataset
    "inference_config__MIN_UNIQUE_FOR_NUMERICAL_FEATURES": hp.choice("MIN_UNIQUE_FOR_NUMERICAL_FEATURES", [0])
}