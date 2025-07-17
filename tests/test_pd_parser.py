from pathlib import Path
from tabutils.prediction import PredictionDataframe
import pytest


 
@pytest.mark.filterwarnings("ignore:Found NAs in")
def test_parsing_works_with_na_in_columns_to_parse():
    path = Path(__file__).parents[1] / "data/pred_dataframe.txt"
    df_pred = PredictionDataframe()
    df_pred.build_from_file(path, parse=True, sep="\t")