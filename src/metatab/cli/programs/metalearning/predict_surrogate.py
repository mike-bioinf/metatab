"""
Program to use the surrogate models to predict on metadata.
Allow to optionally specify a surrogate model to use or a folder of them.
In the last case the loo structure enforced by surrogate and metadata names must be respected. 
We use the surrogate models to predict the z-normalized losses.
The output is a file with the expected and predicted loss for each meta-configuration.
"""

from __future__ import annotations

import sys
import argparse
import joblib
import pandas as pd
from typing import TYPE_CHECKING
from logging import Logger
from pathlib import Path
from metatab.cli.helper import create_logger

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline



def parse_args(args):
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate-input", required=True, help="Path of the folder with the surrogate models. Optionally a file of one surrogate to use.")
    p.add_argument("--metadata-folder", required=True, help="Path of the folder containing the meta-datasets.")
    p.add_argument("--output-file", required=True, help="Name of the output file with the real and predicted losses.")
    p.add_argument("--loss-column", default="z_normalized_loss", help="Name of the meta-dataset column to treat as the y target.")
    p.add_argument("--skip-missing", action="store_true", 
                   help="""Whether to skip the meta-datasets for which there is no surrogate available. 
                   If False a preliminary check is done on the correspondence, and an error raised in case it is not satisfied.""")
    return p.parse_args(args)



def log_program_setting(logger: Logger, pars: dict, is_single_surrogate: bool):
    surrogate_input = pars["surrogate_input"]
    surrogate_log = f"with the surrogate model '{surrogate_input}'" \
        if is_single_surrogate \
        else f"with the surrogate models in folder '{surrogate_input}'"
    logger.debug(
        (
            f"\nLaunching the 'predict_surrogate' program on folder '{pars["metadata_folder"]}'"
            f" {surrogate_log}\n"
        )
    )


def load_metadata(folder: Path) -> dict[str, pd.DataFrame]:
    '''
    Load the metadata and insert them in a dict with keys equal to the basename of the files.
    '''
    dfs = {}
    for file in folder.iterdir():
        if not file.is_file():
            raise ValueError(f"Unexpected folder in metafolder: {file.name}")
        try:
            basename = file.stem
            dfs[basename] = pd.read_csv(file, sep="\t")
        except Exception as e:
            raise ValueError(f"Error in reading the meta-dataset {file.name}") from e
    return dfs



def load_surrogates(surrogate_path: Path, is_single_surrogate: bool) -> Pipeline | dict[str, Pipeline]:
    '''
    Load the surrogate. If "is_single_surrogate" is True then it return the surrogate object (Pipeline) 
    else a dict with file basenames as keys and the surrogate objects (Pipeline) as values.
    '''
    if is_single_surrogate:
        return joblib.load(surrogate_path)
    else:
        surrogates = {}
        for file in surrogate_path.iterdir():
            if not file.is_file():
                raise ValueError(f"Unexpected folder in surrogate folder: {file.name}")
            try:
                basename = file.stem
                surrogates[basename] = joblib.load(file)
            except Exception as e:
                raise ValueError(f"Error in loading surrogate {file.name}") from e
        return surrogates



def check_correspondence_surrogate_metadata(
    surrogates: dict[str, Pipeline], 
    metadata: dict[str, pd.DataFrame]
) -> None:
    '''
    Check reference (original dataset) correspondence between the surrogates and metadata.
    In details check whether the number and names are the same.
    '''
    name_metadata = list(metadata.keys())
    name_surrogates = list(surrogates.keys())
    if len(name_metadata) != len(name_surrogates):
        raise ValueError("The number of surrogates is different from the number of metadata.")
    if set(name_metadata) != set(name_surrogates):
        raise ValueError("Discrepancies found in metadata and surrogate dataset references.")



def main():
    pars = vars(parse_args(sys.argv[1:]))
    surrogate_path = Path(pars["surrogate_input"])
    metadata_path = Path(pars["metadata_folder"])
    output_file = Path(pars["output_file"])

    is_single_surrogate = surrogate_path.is_file()
    output_folder = output_file.parent
    output_folder.mkdir(exist_ok=True, parents=True)

    logger = create_logger(sys.stdout)
    log_program_setting(logger, pars, is_single_surrogate)

    # load
    metadata = load_metadata(metadata_path)
    logger.debug("Loaded metadata.")
    surrogates = load_surrogates(surrogate_path, is_single_surrogate)
    logger.debug("Loaded surrogates.")

    # check and have surrogate as a dict with keys equivalent to metadata
    if is_single_surrogate:
        surrogates = {k:surrogates for k in metadata.keys()}
    elif not pars["skip_missing"]:
        check_correspondence_surrogate_metadata(surrogates, metadata)

    results = []
    for reference in metadata.keys():
        m = metadata[reference]
        # allow to skip dataset for which we have the metadata but not the surrogate
        try:
            s = surrogates[reference]
        except Exception:
            if pars["skip_missing"]:
                logger.debug(f"Skipped {reference} since no surrogate is available.")
                continue
            else:
                raise
        X = m.drop(columns=pars["loss_column"])
        y = m[pars["loss_column"]]
        preds, uncertainty = s.predict(X)
        result = pd.DataFrame({"actual": y, "pred": preds, "uncertainty": uncertainty})
        result.insert(0, "dataset", reference)
        results.append(result)
        logger.debug(f"Inference completed on {reference}.")

    results: pd.DataFrame = pd.concat(results, axis=0, ignore_index=True)
    results.to_csv(pars["output_file"], sep="\t", index=False)
    logger.debug(f"\n Results saved in the dataframe {pars["output_file"]}.")



if __name__ == "__main__":
    main()