from __future__ import annotations

import sys
import joblib
import logging
import requests
from typing import TYPE_CHECKING
from platformdirs import user_cache_path

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from estimators.utils.types import TunableEstimatorType



def download_surrogate_framework(type_estimator: TunableEstimatorType) -> None:
    '''
    Download the surrogate model/framework for the input estimator 
    from the hugging face repo in the user metatab cache directory.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    model = f"surrogate_framework_for_{type_estimator}.joblib"
    cache_dir = user_cache_path("metatab")
    cache_dir.mkdir(exist_ok=True)
    logger.info(f"Attempting to download the surrogate model from HuggingFace in: {cache_dir}")

    dest = cache_dir / model
    src = f"https://huggingface.co/piupo/metatab_surrogate_pipelines/resolve/main/models/{model}"
    r = requests.get(src)

    try:
        r.raise_for_status()
    except Exception as e:
        raise ValueError(f"The model request process failed with the following error {e}")

    ## TODO: add a progress bar?
    with open(dest, "wb") as f:
        f.write(r.content)

    logger.info("Download completed!")



def query_surrogate_framework(type_estimator: TunableEstimatorType) -> Pipeline:
    '''Retrieve the fitted surrogate framework for the input type_estimator'''
    surrogate_model_path = user_cache_path("metatab") / f"surrogate_framework_for_{type_estimator}.joblib"
    if not surrogate_model_path.exists():
        download_surrogate_framework(type_estimator)
    return joblib.load(surrogate_model_path)