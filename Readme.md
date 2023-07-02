# Target Permutation Importances

## Overview
This method aims at lower the feature attribution due to the variance of a feature.
If a feature is important after the target vector is shuffled, it is fitting to noise.

By default, this package 

1. Fit the given model class on the given dataset M times to compute the mean actual feature importances ($A$).
2. Fit the given model class on the given dataset with shuffled targets for N times to compute mean random feature importances ($R$).
3. Compute the final importances by either $A - R$ or $A / (MinMaxScale(R) + 1)$

Not to be confused with [sklearn.inspection.permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance),
this sklearn method is about feature permutation instead of target permutation.

This method were originally proposed/implemented by:
- [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)


## Basic Usage

```python
# Import the function
from target_permutation_importances import compute

# Prepare a dataset
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = load_breast_cancer()

# Compute permutation importances with default settings
result_df = compute(
    # RandomForestClassifier, XGBClassifier, CatBoostClassifier, LGBMClassifier...
    model_cls=RandomForestClassifier,
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
Instead of calling `compute` this package also expose `generic_compute` to allow customization.
Read `target_permutation_importances.__init__` for details.


## Feature Selection Examples

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

For binary classification task, `sklearn.metrics.f1_score` is used for evaluation. For regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The downloaded datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%.
Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.

## Kaggle Competitions
Many Kaggle Competition top solutions involve using this method, here are some examples

| Year | Competition                                                                                                                  | Medal | Link                                                                                                                                        |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play) | Gold  | [3rd place solution](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420235)                      |
| 2019 | [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/overview)    | Gold  | [16th place solution]([-play/discussion/420235](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82166)) |
| 2018 | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)                            | Gold  | [10th place solution](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64598)                                        |


## Development Setup and Contribution Guide
**Python Version**

You can find the suggested development Python version in `.python-version`.
You might consider setting up `Pyenv` if you want to have multiple Python versions in your machine.

**Python packages**

This repository is setup with `Poetry`. If you are not familiar with Poetry, you can find packages requirements are listed in `pyproject.toml`. 
Otherwise, you can just set up with `poetry install`

**Run Benchmarks**

To run benchmark locally on your machine, run `make run_tabular_benchmark` or `python -m benchmarks.run_tabular_benchmark`

**Make Changes**

Following the [Make Changes Guide from Github](https://github.com/github/docs/blob/main/CONTRIBUTING.md#make-changes)
Before committing or merging, please run the linters defined in `make lint` and the tests defined in `make test`

