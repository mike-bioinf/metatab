# Metatab

A comprehensive classification framework for microbial taxonomic profiles.

## Features
- **Wide Model Library**: Includes 9 models spanning trees, deep learning methods, and AutoGluon
- **Tailored HPO Search Spaces**: Offers hyperparameter spaces tailored for microbial profiles
- **Flexible HPO Optimization**: Supports different hyperparameter optimization algorithms
- **Default, Tune and Ensemble Model Versions**: Each model supports default parameters, HPO tuned and ensembled versions
- **Metalearning options**: Provides metalearning capabilities to guide HPO
- **Hierarchical Ensembling**: Enables building ensembles of model ensembles
- **Multiple CLI Fitting Strategies**: Supports whole-dataset, holdout, and cross-validation fitting strategies via CLI on custom datasets
- **Basic Preprocessing Support**: Supports a limited suite of data preprocessing options


## Installation

```bash
# Create conda environment
conda create --name metatab
conda activate metatab

# Clone the repository
git clone https://github.com/mike-bioinf/metatab.git

# Install metatab in development mode
pip install -e metatab
```


## Quick Start

### CLI
Metatab provides three main commands:
- metatab-fit
- metatab-resample
- metatab-predict

Each command serves a different purpose and exposes a different subcommand structure.


### metatab-resample
metatab-resample is used to fit and evaluate estimators on a dataset using a resampling strategy.
It supports two levels of subcommands:

- Resampling strategy, defines how the data is split: cv or holdout.

- Estimator mode, defines how estimators are trained within the chosen resampling strategy: default, tune, ensemble, family-ensemble and autogluon.

In practice, a metatab-resample command always selects one resampling strategy followed by one estimator mode (e.g. metatab-resample cv tune).

### metatab-fit
metatab-fit is used to fit an estimator on the entire dataset, without applying any resampling strategy.
It supports only estimator-mode subcommands: default, tune, ensemble, family-ensemble and autogluon.
This command is typically used when the goal is to train a model for later deployment or inference.


### metatab-predict
metatab-predict has no subcommands.
It is used to evaluate a previously fitted estimator on new data. 
The pickled estimator needed in input can be obtained from:
- metatab-fit
- metatab-resample, when run with the --save-estimators option enabled.


### Quick usage guide
- Use metatab-resample to train and evaluate models under CV or holdout resampling.
- Use metatab-fit to train a model on the full dataset.
- Use metatab-predict to apply a trained model to new, external data.

```bash
# Fit a model on a dataset
metatab-fit default \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --estimator random_forest \
    --preprocessing estimator_default \
    --seed 42 \
    --nthreads 1 \
    --create-outdir

# Fit a tuned model on a dataset
metatab-fit tune \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --estimator es_lgbm \
    --validation-set-size 0.3 \
    --early-stop-rounds 10 \
    --preprocessing estimator_default \
    --tune-algo random \
    --tune-n-iter 10 \
    --tune-n-cv-repeats 1 \
    --tune-n-cv-folds 5 \
    --nthreads 1 \
    --create-outdir


# Fit an ensembled model on a dataset using metatab metalearning capabilities
metatab-fit ensemble \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --estimator extra_trees \
    --preprocessing estimator_default \
    --ensemble-algo meta \
    --ensemble-n-members 8 \
    --ensemble-time-limit 600 \
    --nthreads 1 \
    --create-outdir


# Fit a hierarchical ensemble of all classifiers on a dataset
metatab-fit family-ensemble \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --ensemble-configuration "all_random_8" \
    --nthreads 1 \
    --create-outdir

# Use a fitted model to obtain predictions and performance metrics on a second dataset
metatab-predict \
    --file-estimator "path/fitted/estimator_file" \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --x-uniform \
    --create-outdir

# Fit autogluon on a dataset
metatab-fit autogluon \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --preset extreme_quality \
    --time-limit 600 \
    --eval-metric log_loss \
    --nthreads 1 \
    --ngpus 1 \
    --create-outdir

# Fit ensemble models on a dataset in a cross-validation procedure
metatab-resample cv tune \
    --input-data "path/to/your/data" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "your_target_column" \
    --estimator extra_trees \
    --preprocessing estimator_default \
    --ensemble-algo random \
    --ensemble-n-members 8 \
    --ensemble-time-limit 600 \
    --seed-splitter 42 \
    --seed-estimator 42 \
    --n-cv-repeats 1 \
    --n-cv-folds 5 \
    --nthreads 1 \
    --create-outdir

# consult help pages for detailed info
metatab-resample cv tune --help
```



### Python API
Metatab exposes meta tuned and ensembled classifiers and the hierachical (or family) ensembler constructor.

```python
from metatab import MetaTuneRandomForestClassifier, FamilyEnsembleEstimator
from metatab.ensemble.configuration import UserEnsembleConfiguration, CollectionUserEnsembleConfiguration

## zero-shot meta-tuning
meta_tuned_rf = MetaTuneRandomForestClassifier(n_iter=1)
## use your data
# meta_tuned_rf.fit(X_train, y_train)
# pred_proba = meta_tuned_rf.predict_proba(X_test)


## Building an hierarchical ensemble

# 0. specify the inner-ensemble configurations
rf_0 = UserEnsembleConfiguration(
    name="ens_rf_0",
    algo="random",
    n_members=2,
    estimator="random_forest",
    preprocessing="estimator_default",
    tune_space="default",
    early_stop_on_validation_set=False,
)

es_xgb_0 = UserEnsembleConfiguration(
    name="ens_esxgb_0",
    algo="meta",
    n_members=4,
    estimator="es_xgb",
    preprocessing="estimator_default",
    tune_space="default",
    early_stop_on_validation_set=True,
    validation_set_size=0.2
)

# 1. Merge the configurations into a single one
family_ensemble_configuration = CollectionUserEnsembleConfiguration([rf_0, es_xgb_0])

# 2. build the hierarchical ensemble
family_ensemble = FamilyEnsembleEstimator(
    name="fam_ens",
    configuration=family_ensemble_configuration,
    save_path="path/output/folder"
)

## use your data
# family_ensemble.fit(X_train, y_train)
# pred_proba = family_ensemble.predict_proba(X_test)
```



## Available Models

| Model | Default | Tuned (HPO) | Ensembled (HPO) |
|-------|---------|-------------|----------------|
| Extra Trees | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ |
| LightGBM | ✅ | ✅ | ✅ |
| CatBoost | ✅ | ✅ | ✅ |
| TabM | ✅ | ✅ | ✅ |
| RealMLP | ✅ | ✅ | ✅ |
| TabPFN v2 | ✅ | ✅ | ✅ |
| AutoGluon | ✅ | N/A* | N/A* |

*N/A: Model handles tuning and ensembling internally.


## Authors

- **Michele Avagliano** - University of Naples Federico II
- **Edoardo Pasolli** - University of Naples Federico II

## License

MIT License - see LICENSE file for details.

## Citation

If you use Metatab in your research, please cite:

```bibtex
@software{metatab,
  author = {Avagliano, Michele and Pasolli, Edoardo},
  title = {Metatab: A Classification Benchmark Framework for Microbial Taxonomic Profiles},
  year = {2026},
  url = {https://github.com/mike-bioinf/metatab}
}
```