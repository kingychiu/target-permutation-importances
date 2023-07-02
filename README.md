# Target Permutation Importances

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/kingychiu/target-permutation-importances)
[![image](https://img.shields.io/pypi/v/target-permutation-importances.svg)](https://pypi.python.org/pypi/target-permutation-importances)
[![image](https://img.shields.io/pypi/pyversions/target-permutation-importances.svg)](https://pypi.python.org/pypi/target-permutation-importances)
[![Actions status](https://github.com/kingychiu/target-permutation-importances/workflows/CI/badge.svg)](https://github.com/kingychiu/target-permutation-importances/actions/workflows/main.yaml)



## Overview
This method aims to lower the feature attribution due to a feature's variance.
If a feature shows high importance to a model after the target vector is shuffled, it fits the noise.

Overall, this package 

1. Fit the given model class $M$ times to get $M$ actual feature importances of feature f: $A_f = [a_{f_1},a_{f_2}...a_{f_M}]$.
2. Fit the given model class with shuffled targets for $N$ times to get $N$ feature random importances: $R_f = [r_{f_1},r_{f_2}...r_{f_N}]$.
3. Compute the final importances of a feature $f$ by various methods, such as:
    - $A_f$ - $R_f$
    - $A_f$ - ($R_f + 1)$

Not to be confused with [sklearn.inspection.permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance),
this sklearn method is about feature permutation instead of target permutation.

This method were originally proposed/implemented by:
- [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)


## Install

```
pip install target-permutation-importances
```
or
```
poetry add target-permutation-importances
```

## Basic Usage

```python
# Import the function
from target_permutation_importances import compute

# Prepare a dataset
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data = load_breast_cancer()

# Convert to a pandas dataframe
Xpd = pd.DataFrame(data.data, columns=data.feature_names)

# Compute permutation importances with default settings
result_df = compute(
    model_cls=RandomForestClassifier, # Or other models
    model_cls_params={ # The params for the model class construction
        "n_estimators": 1,
    },
    model_fit_params={}, # The params for model.fit
    X=Xpd,
    y=data.target,
    num_actual_runs=2,
    num_random_runs=10,
)
```

You can find more detailed examples in the "Feature Selection Examples" section.

## Advance Usage / Customization
This package exposes `generic_compute` to allow customization.
Read [`target_permutation_importances.__init__.py`](target_permutation_importances/__init__.py) for details.


## Feature Selection Examples
TODO

## Benchmarks

Benchmark has been done with some tabular datasets from the [Tabular data learning benchmark](https://github.com/LeoGrin/tabular-benchmark/tree/main). It is also
hosted on [Hugging Face](https://huggingface.co/datasets/inria-soda/tabular-benchmark).

The following models with their default params are used in the benchmark:
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.RandomForestRegressor`
- `xgboost.XGBClassifier`
- `xgboost.XGBRegressor`
- `catboost.CatBoostClassifier`
- `catboost.CatBoostRegressor`
- `lightgbm.LGBMClassifier`
- `lightgbm.LGBMRegressor`

For the binary classification task, `sklearn.metrics.f1_score` is used for evaluation. For the regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The downloaded datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%.
Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.


Raw result data are in [`benchmarks/results/tabular_benchmark.csv`](benchmarks/results/tabular_benchmark.csv).

## Kaggle Competitions
Many Kaggle Competition top solutions involve this method, here are some examples

| Year | Competition                                                                                                                  | Medal | Link                                                                                                                                        |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play) | Gold  | [3rd place solution](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420235)                      |
| 2019 | [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/overview)    | Gold  | [16th place solution]([-play/discussion/420235](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82166)) |
| 2018 | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)                            | Gold  | [10th place solution](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64598)                                        |


## Development Setup and Contribution Guide
**Python Version**

You can find the suggested development Python version in `.python-version`.
You might consider setting up `Pyenv` if you want to have multiple Python versions on your machine.

**Python packages**

This repository is setup with `Poetry`. If you are not familiar with Poetry, you can find package requirements listed in `pyproject.toml`. 
Otherwise, you can just set it up with `poetry install`

**Run Benchmarks**

To run the benchmark locally on your machine, run `make run_tabular_benchmark` or `python -m benchmarks.run_tabular_benchmark`

**Make Changes**

Following the [Make Changes Guide from Github](https://github.com/github/docs/blob/main/CONTRIBUTING.md#make-changes)
Before committing or merging, please run the linters defined in `make lint` and the tests defined in `make test`

