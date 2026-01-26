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
The pickled estimator needed by this program can be obtained from:
- metatab-fit
- metatab-resample, when run with the --save-estimators option enabled.


### Quick usage guide
- Use metatab-resample to train and evaluate models under CV or holdout resampling.
- Use metatab-fit to train a model on the full dataset.
- Use metatab-predict to use a trained model to new, external data.


### Note on metatab-resample and metatab-predict output
The main output of the metatab-resample and metatab-predict programs is a txt file referred as "pred_dataframe".
This is a tab-separated file containing estimator predictions and performance metrics info, among the others.
Some fields (such as predictions) are stored as strings that can be decoded back into the original numpy array objects.
This conversion is executed by the "PredictionDataframe" python class (see Python API below), 
which provides a convenient way to load and parse these files into pandas DataFrames.
By default, the same information is also written in the non-encoded form in additional text files. 
The creation of these additional redundant outputs can be avoided via the "--disable-additional-txt-output" flag.

```bash
example_dataset_path=$(metatab-get-example-data-path)

# Fit a model on a dataset
metatab-fit default \
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
    --estimator random_forest \
    --preprocessing estimator_default \
    --seed 42 \
    --nthreads 1 \
    --create-outdir

# Fit a tuned model on a dataset
metatab-fit tune \
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
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
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
    --estimator extra_trees \
    --preprocessing estimator_default \
    --ensemble-algo meta \
    --ensemble-n-members 8 \
    --ensemble-time-limit 600 \
    --nthreads 1 \
    --create-outdir


# Fit a hierarchical ensemble of all classifiers on a dataset
metatab-fit family-ensemble \
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
    --ensemble-configuration "all_random_8" \
    --nthreads 1 \
    --create-outdir

# Use a fitted model to obtain predictions and performance metrics on a second dataset
metatab-predict \
    --file-estimator "path/fitted/estimator_file" \
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
    --x-uniform \
    --create-outdir

# Fit autogluon on a dataset
metatab-fit autogluon \
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
    --preset extreme_quality \
    --time-limit 600 \
    --eval-metric log_loss \
    --nthreads 1 \
    --ngpus 1 \
    --create-outdir

# Fit ensemble models on a dataset in a cross-validation procedure
metatab-resample cv ensemble \
    --input-data "${example_dataset_path}" \
    --output-dir "path/of/your/output-directory" \
    --input-mode df \
    --target-feature "Group" \
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
- Metatab exposes meta tuned and ensembled classifiers and the hierachical (or family) ensembler constructor.
- The PredictionDataframe class can be used to easily load and parse the pred_dataframe* files generated through the CLI API into a pandas DataFrame.

```python
from sklearn.model_selection import train_test_split
from metatab import MetaTuneRandomForestClassifier, FamilyEnsembleEstimator, get_example_data
from metatab.ensemble.configuration import UserEnsembleConfiguration, CollectionUserEnsembleConfiguration
from metatab.metatab_utils.prediction import PredictionDataframe


# get data and split in train and test sets
data = get_example_data()
X = data.drop(columns="Group")
y = data["Group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0, stratify=y)


## zero-shot meta-tuning
meta_tuned_rf = MetaTuneRandomForestClassifier(n_iter=1)
meta_tuned_rf.fit(X_train, y_train)
pred_proba = meta_tuned_rf.predict_proba(X_test)


## Building an hierarchical ensemble
# 0. Specify the inner-ensemble configurations
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
    algo="random",
    n_members=2,
    estimator="es_xgb",
    preprocessing="estimator_default",
    tune_space="default",
    early_stop_on_validation_set=True,
    validation_set_size=0.2
)

# 1. Merge the configurations into a single one
family_ensemble_configuration = CollectionUserEnsembleConfiguration([rf_0, es_xgb_0])

# 2. Build the hierarchical ensemble
family_ensemble = FamilyEnsembleEstimator(
    name="fam_ens",
    configuration=family_ensemble_configuration,
    save_path="output" ## put your folder output path (it will be created automatically if not existent)
)

# 3. Fit and then predict
family_ensemble.fit(X_train, y_train)
pred_proba = family_ensemble.predict_proba(X_test)


### load the "pred_dataframe" files generated with the CLI API into pandas dataframes
pdf = PredictionDataframe()
# out_df = pdf.build_from_file(file="pred_dataframe.txt", sep="\t").get_df()
# out_df = pdf.build_from_folder(folder="results", glob_pattern="pred_dataframe*", recursive=True, sep="\t").get_df()
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