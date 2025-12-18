from __future__ import annotations

import sys
import json
import joblib
import logging
from typing import TYPE_CHECKING
from importlib.metadata import version
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from metatab.estimators.utils.types import TunableEstimatorType



def download_surrogate_framework(type_estimator: TunableEstimatorType) -> str:
    '''
    Download the surrogate model/framework for the input estimator 
    from the hugging face repo in the user hf cache directory.
    Returns the user system model filepath.
    '''
    logger = logging.getLogger(__name__)
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
        subfolder=resolve_surrogate_models_folder()
    )

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