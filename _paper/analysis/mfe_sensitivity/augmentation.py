"""
Program to compute surrogate sensitivity to metafeatures groups using a single dataset and a pertubation approach.
In detail we pertube the dataset in input and evaluate surrogate sensitivity to metafeatures
as difference between predictions on original and pertubed dataset.
The output is a txt table that reports for every metagroup and iteration/pertubation the sensitivity score
as the mean across samples of absolute differences between base surrogate predictions and pertubed ones.
"""
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any
from collections import defaultdict
from metatab.cli.helper import create_logger
from metatab.metatab_utils.data_loader import DataLoader
from metatab.metatab_utils.general import select_level_from_columns
from metatab.hp_search.point_corrector import PointCorrector
from metatab.metalearning.metafeatures import CustomMFE
from metatab.metalearning.metadata_generator import MetadataGenerator
from metatab.metalearning.sampler import HyperoptRandomSampler
from metatab.estimators.params.utils import pick_estimator_tune_space

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--output-file", required=True, help="Output filepath")
    p.add_argument("--surrogate-model", required=True, help="Surrogate model to use, aka joblib serialized file.")
    p.add_argument("--estimator", required=True, help="Estimator type needed to pick the tune space")
    p.add_argument("--dataset", required=True, help="Dataset filepath from which get metafeatures. Must be a tab separated file.")
    p.add_argument("--target-feature", required=True, help="Target feature to exlude from the dataset")
    p.add_argument("--n-iterations", type=int, default=100, help="Number of dataset pertubations to evaluate.")
    p.add_argument("--n-points", type=int, default=1500, help="Number of hps points to draw.")
    p.add_argument("--meta-seed", type=int, default=42, help="Seed used in the candidate points drawing process")
    p.add_argument("--seed", type=int, default=0, help="Seed controlling pertubations randomness")
    return p.parse_args(args)



def pertube_data(
    X: pd.DataFrame, 
    y: pd.Series,
    class_0_value: Any = 0,
    class_1_value: Any = 1,
    class_0_fractions: list[float] = [0.6, 0.8, 1, 1.2, 1.4],
    class_1_fractions: list[float] = [0.6, 0.8, 1, 1.2, 1.4],
    feature_fractions: list[float] = [0.6, 0.8, 1, 1.2, 1.4],
    n_iterations: int = 100,
    seed: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    '''
    Utility to pertube the data via sub and over sampling of features and rows.
    The row sub/over sampling is guided by class, in the sense that we act
    on the specific class. Here we allow only for 2 different classes.

    Parameters:
        X: Dataframe to pertube.
        y: Series with target classes. Must contain 2 different classes.
        class_0_value: Value of class_0.
        class_1_value: Value of class_1.
        class_0_fractions: List of sampling fractions used for class_0
        class_1_fractions: List of sampling fractions used for class_1
        feature_fractions: List of sampling fraction used for features
        n_iterations: Number of pertubations applied.
        seed: Control process randomness.

    Returns 3 lists of class_0, class_1 and feature indexes arrays 
    to select to pertube the original data.
    '''
    if (
        set(class_0_fractions) == {1}
        and set(class_1_fractions) == {1}
        and set(feature_fractions) == {1}
    ):
        raise ValueError("All fractions are 1 → no perturbation possible")

    rng = np.random.default_rng(seed)
    y_size = y.size

    feature_idx = np.arange(X.shape[1])
    class_0_idx = np.arange(y_size)[y == class_0_value]
    class_1_idx = np.arange(y_size)[y == class_1_value]

    if class_0_idx.size == 0:
        raise ValueError(f"Class 0 value '{class_0_value}' not found in y")
    if class_1_idx.size == 0:
        raise ValueError(f"Class 1 value '{class_1_value}' not found in y")

    selected_class_0_idx = []
    selected_class_1_idx = []
    selected_feature_idx = []
    n_iter_done = 0

    while n_iter_done < n_iterations:
        class_0_fraction = rng.choice(class_0_fractions)
        class_1_fraction = rng.choice(class_1_fractions)
        feature_fraction = rng.choice(feature_fractions)

        # skip the combination that does not change the data
        if class_0_fraction == 1 and class_1_fraction == 1 and feature_fraction == 1:
            continue

        selected_feature_idx.append(
            rng.choice(
                a=feature_idx, 
                size=int(feature_fraction * feature_idx.size),
                replace=feature_fraction > 1
            )
        )

        selected_class_0_idx.append(
            rng.choice(
                a=class_0_idx, 
                size=int(class_0_fraction * class_0_idx.size),
                replace=class_0_fraction > 1
            )
        )

        selected_class_1_idx.append(
            rng.choice(
                a=class_1_idx,
                size=int(class_1_fraction * class_1_idx.size),
                replace=class_1_fraction > 1
            )
        )

        n_iter_done+=1

    return selected_class_0_idx, selected_class_1_idx, selected_feature_idx



def main():
    pars = vars(parse_args(sys.argv[1:]))
    logger = create_logger(sys.stdout)
    
    dl = DataLoader()
    dl.load_df_mode(pars["dataset"], pars["target_feature"], load_as="generic", sep="\t")
    X, y = dl.X, dl.y
    dataset_name = dl.generic_dataset_name
    
    logger.info(f"Using {pars["surrogate_model"]} surrogate model on {dataset_name}")

    tune_space = pick_estimator_tune_space(pars["estimator"], space="c0")
    surrogate_model: Pipeline = joblib.load(pars["surrogate_model"])

    metadata_generator = MetadataGenerator(
        sampler=HyperoptRandomSampler(),
        point_corrector=PointCorrector(),
        mfe=CustomMFE()
    )

    # generate original metadata
    metadata_generator.fit(X, y, hp_space=tune_space, seed=pars["meta_seed"])
    metadata, _ = metadata_generator.generate(n_points=pars["n_points"], set_metagroups_in_index=True)
    metadata[("preprocessing", "preprocessing")] = "density_filter" if pars["estimator"] == "tabpfn" else "base"
    ori_pred, _ = surrogate_model.predict(select_level_from_columns(metadata, 1))

    map_group_sensitivity = defaultdict(list)
    
    for i, (class_0_idx, class_1_idx, col_idx) in enumerate(zip(
        *pertube_data(X, y, n_iterations=pars["n_iterations"], seed=pars["seed"])
    )):
        logger.info(f"Starting iteration {i}")        
        
        X_pertubed_features = X.iloc[:, col_idx]
        X_pertubed_class_0 = X_pertubed_features.iloc[class_0_idx, :]
        X_pertubed_class_1 = X_pertubed_features.iloc[class_1_idx, :]
        X_pertubed = pd.concat([X_pertubed_class_0, X_pertubed_class_1], axis=0, ignore_index=True)
        y_pertubed = pd.concat([y.iloc[class_0_idx], y.iloc[class_1_idx]], ignore_index=True)

        metadata_generator.fit(X_pertubed, y_pertubed, hp_space=tune_space, seed=pars["meta_seed"])
        metadata_from_pertubation, _ = metadata_generator.generate(n_points=pars["n_points"], set_metagroups_in_index=True)
        metadata_from_pertubation[("preprocessing", "preprocessing")] = "density_filter" if pars["estimator"] == "tabpfn" else "base"

        for mfe_group in metadata.columns.unique(0):
            if mfe_group in ["hps", "preprocessing"]: continue # skip non mfe groups
            mask = metadata.columns.get_level_values(0) == mfe_group
            metadata_pertubed = metadata.copy()
            metadata_pertubed.loc[:, mask] = metadata_from_pertubation.loc[:, mask]
            pertubed_pred, _ = surrogate_model.predict(select_level_from_columns(metadata_pertubed, 1))
            sensitivity_score = np.mean(np.abs(ori_pred - pertubed_pred))
            map_group_sensitivity[mfe_group].append(sensitivity_score)

    
    df_out = pd.DataFrame(map_group_sensitivity)
    mfe_group_cols = df_out.columns.to_list()
    df_out["dataset"] = dataset_name
    df_out["n_iteration"] = [f"iteration_{i}" for i in range(pars["n_iterations"])]

    df_out = df_out.melt(
        id_vars=["dataset", "n_iteration"],
        value_vars=mfe_group_cols,
        var_name="mfe_group",
        value_name="iteration"
    )

    df_out.to_csv(pars["output_file"], sep="\t")
    logger.info(f"Output generated at '{pars["output_file"]}'")
            



if __name__ == "__main__":
    main()