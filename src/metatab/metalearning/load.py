from __future__ import annotations

import sys
import json
import joblib
import logging
from typing import TYPE_CHECKING
from importlib.metadata import version
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from huggingface_hub import hf_hub_download, try_to_load_from_cache

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.utils.types import TunableEstimatorType



def download_surrogate_framework(type_estimator: TunableEstimatorType) -> str:
    '''
    Download the surrogate model/framework for the input estimator 
    from the hugging face repo in the user hf cache directory.
    Returns the user system model filepath.
    '''
    models_subfolder = resolve_surrogate_models_folder()

    # we use this mechanisms based on the cache existence to decide when logging.
    # we call in every case the "hf_hub_download" function as main hf function to do the job.
    # the result of this call is a string if cache exists else None or _CACHED_NO_EXIST (not a string)
    cache_existence: str | None = try_to_load_from_cache(
        repo_id="piupo/metatab_surrogate_pipelines",
        filename=f"{models_subfolder}/surrogate_framework_for_{type_estimator}.joblib",
    )

    if not isinstance(cache_existence, str):
        logger = logging.getLogger(__name__ + "_" + type_estimator)
        logger.setLevel(logging.INFO)
        logger.propagate=False
        logger.handlers.clear()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info(f"Attempting to download the surrogate model from HuggingFace.")

    model_filepath = hf_hub_download(
        repo_id="piupo/metatab_surrogate_pipelines",
        filename=f"surrogate_framework_for_{type_estimator}.joblib",
        subfolder=models_subfolder
    )

    if not isinstance(cache_existence, str):
        logger.info("Download completed!")
    
    return model_filepath



def resolve_surrogate_models_folder() -> str:
    '''
    Returns the name of the subfolder containing the 
    surrogate models compatible with the user metatab version.
    '''
    # we download the latest manifest version
    path_manifest = hf_hub_download(
        repo_id="piupo/metatab_surrogate_pipelines",
        filename="manifest.json"
    )

    with open(path_manifest, "r") as f:
        manifest: dict = json.load(f)

    package_version = Version(version("metatab"))
    
    for _, dict_info in manifest.items():
        if package_version in SpecifierSet(dict_info["package_versions"]):
            return dict_info["models_subpackage"]

    raise ValueError(
        "No compatible surrogate model found for this metatab version."
        " Please report this issue on GitHub."
    )



def query_surrogate_framework(type_estimator: TunableEstimatorType) -> Pipeline:
    '''Retrieve the fitted surrogate framework for the input type_estimator'''
    return joblib.load(download_surrogate_framework(type_estimator))