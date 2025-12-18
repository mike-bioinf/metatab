import numpy as np
from pathlib import Path
from metatab.metatab_utils.prediction import PredictionDataframe



def test_parsing_works():
    path = Path(__file__).parent / "data/pred_dataframe.txt"
    df_pred = PredictionDataframe()
    df_pred.build_from_file(path, sep="\t")
    array = df_pred.df["pred_proba"].iloc[0]
    assert isinstance(array, np.ndarray), "The parse capabilities of pred_dataframe are not working."
    assert array.ndim == 2, "The parse capabilities of pred_dataframe are not working."