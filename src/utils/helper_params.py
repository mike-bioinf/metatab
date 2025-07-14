from pathlib import Path


def adjust_io_paths_(pars: dict) -> None:
    '''
    Convert paths to Path objects.
    The function works in place.
    '''
    pars["input_path"] = Path(pars["input_path"])
    pars["output_path"] = Path(pars["output_path"])