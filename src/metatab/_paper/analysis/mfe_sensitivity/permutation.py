"""
Program to compute surrogate sensitivity to metafeatures groups over all datasets using a permutational approach.
The core idea here is to pertube metafeatures groups belonging to different datasets via permutations.
Applying different permutations in combinations to different hyperparameters point we end up with a 
sensitivity distribution that is summarized via central indices (mean). 
These provides an interpretable measure of sensitivity.
The program returns a txt file.
"""
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from metatab.cli.helper import create_logger
from metatab.metalearning.metafeatures import CustomMFE
from metatab.metalearning.sampler import HyperoptRandomSampler
from metatab.metalearning.feature_sensitivity import compute_feature_sensitivity_map
from metatab.estimators.params.utils import pick_estimator_tune_space

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--output-file", required=True, help="Output filepath. Must have the '.txt' extension.")
    p.add_argument("--surrogate-model", required=True, help="Surrogate model to use, aka joblib serialized file.")
    p.add_argument("--estimator", required=True, help="Estimator type needed to pick the tune space")
    p.add_argument("--datasets-folder", required=True, help="Folder with datasets from which get metafeatures. They must be tab separated files.")
    p.add_argument("--target-feature", required=True, help="Target feature.")
    p.add_argument("--n-points", type=int, default=100, help="Number of hps points.")
    p.add_argument("--n-permutations", type=int, default=100, help="Number of permutations.")
    p.add_argument("--seed", type=int, default=0, help="Seed controlling hps and permutation randomness")
    return p.parse_args(args)



def main():
    pars = vars(parse_args(sys.argv[1:]))
    logger = create_logger(sys.stdout)
    logger.info(f"Using {pars["surrogate_model"]} surrogate model on {pars["datasets_folder"]}")

    dfs = [pd.read_csv(file, sep="\t") for file in Path(pars["datasets_folder"]).iterdir()]
    logger.info(f"Loaded {len(dfs)} datasets in memory.")

    tune_space = pick_estimator_tune_space(pars["estimator"], space="c0")
    surrogate_model: Pipeline = joblib.load(pars["surrogate_model"])
    hp_sampler = HyperoptRandomSampler(follow_hyperopt_fmin=False)
    target_feature = pars["target_feature"]
    mfe = CustomMFE()

    # extract metafeatures
    dfs_mfs = []
    for df in dfs:
        mfs, groups = mfe.fit(df.drop(columns=target_feature), df[target_feature]).extract()
        dfs_mfs.append(mfs)

    # create df metafeatures
    df_mfs = pd.DataFrame(dfs_mfs)
    df_mfs.columns = pd.MultiIndex.from_arrays([groups, df_mfs.columns], names=["group", "feature"])
    
    # add preprocessing info 
    df_mfs[("preprocessing", "preprocessing")] = "density_filter" if pars["estimator"] == "tabpfn" else "base"

    # clean datasets from memory
    del dfs_mfs
    del dfs

    logger.info("Extracted metafeatures from datasets.")

    rng_permutations = np.random.default_rng(pars["seed"])
    map_sensitivity = {}

    for i, hp_point in enumerate(hp_sampler.fit(tune_space, pars["seed"]).sample_points(pars["n_points"])):
        df_point = pd.DataFrame([hp_point] * df_mfs.shape[0])
        df_point.columns = pd.MultiIndex.from_arrays([["hps"] * df_point.shape[1], df_point.columns])
        df_point_mfs = pd.concat([df_point, df_mfs], axis=1)

        map_sensitivity[f"point_{i}"] = compute_feature_sensitivity_map(
            model=surrogate_model,
            X=df_point_mfs,
            column_level_groups=0,
            column_level_prediction=1,
            exclude_groups=["preprocessing", "hps"],
            n_permutations=pars["n_permutations"],
            seed=rng_permutations.integers(0, 2**32-1)
        )

        logger.info(f"Computed group-sensitivity map for point {i}")
    
    dfs_map = []
    for point, map in map_sensitivity.items():
        df_map = pd.DataFrame(map)
        df_map["number_hp_point"] = point
        dfs_map.append(df_map)
    
    df_sensitivity = pd.concat(dfs_map, axis=0, ignore_index=True)
    df_sensitivity.to_csv(pars["output_file"], sep="\t", index=False)
    logger.info(f"Output saved at {pars["output_file"]}")



if __name__ == "__main__":
    main()