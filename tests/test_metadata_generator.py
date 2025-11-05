import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from hp_search.point_corrector import PointCorrector
from metalearning.sampler import HyperoptRandomSampler
from metalearning.generator import MetadataGenerator
from metalearning.metafeatures import CustomMFE
from estimators.params import TuningParams




def test_that_metadata_generator_works():
    generator = MetadataGenerator(
        sampler=HyperoptRandomSampler(),
        point_corrector=PointCorrector(),
        mfe=CustomMFE(seed=0)
    )
    
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)

    metadata, points = generator.fit(X, y, TuningParams.LGMB_C0, seed=0).generate(
        n_points=5, 
        set_metagroups_in_index=True
    )

    # check same number of points between the 2 info
    assert metadata.shape[0] == len(points), "MetadataGeneretor produce metadata and list of points of different size."
    
    # check presence multiindex
    assert isinstance(metadata.columns, pd.MultiIndex), "MetadataGenerator does not produce multiindexed metadata."
    
    # check that all hps are correctly labelled
    mask_hps = metadata.columns.get_level_values("group") == "hps"
    hps = metadata.loc[:, mask_hps]
    assert hps.shape[1] == len(points[0]), "MetadataGenerator incorrectly labels hps in metadata"
    assert np.all(hps.columns.get_level_values(-1).to_numpy() == np.array(list(points[0].keys()))), "Different hps between dataframe and points."

    # check alignment in values betweeen dataframe rows and points
    hps_index = hps.columns.get_level_values(-1)
    for i in range(len(points)):
        hps_row = pd.Series(hps.iloc[i, :].to_numpy(), index=hps_index).to_dict()
        assert (hps_row == points[i]), "MetadataGenerator produces disaligned info between df rows and list of points"