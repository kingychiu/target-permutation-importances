# Target Permutation Importances (Null Importances)

[![image](https://img.shields.io/pypi/v/target-permutation-importances.svg)](https://pypi.python.org/pypi/target-permutation-importances)
[![Downloads](https://static.pepy.tech/badge/target-permutation-importances)](https://pepy.tech/project/target-permutation-importances)
[![image](https://img.shields.io/pypi/pyversions/target-permutation-importances.svg)](https://pypi.python.org/pypi/target-permutation-importances)
[![Actions status](https://github.com/kingychiu/target-permutation-importances/workflows/CI/badge.svg)](https://github.com/kingychiu/target-permutation-importances/actions/workflows/main.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/kingychiu/target-permutation-importances)

[[Source]](https://github.com/kingychiu/target-permutation-importances/)
[[Bug Report]](https://github.com/kingychiu/target-permutation-importances/issues/)
[[Documentation]](https://target-permutation-importances.readthedocs.io/en/latest/)
[[API Reference]](https://target-permutation-importances.readthedocs.io/en/latest/reference/)

## Overview
This method aims to lower the feature attribution due to a feature's variance.
If a feature shows high importance to a model after the target vector is shuffled, it fits the noise.

Overall, this package 

1. Fit the given model class $M$ times with different model's `random_state` to get $M$ actual feature importances of feature f: $A_f = [a_{f_1},a_{f_2}...a_{f_M}]$.
2. Fit the given model class with different model's `random_state` and **shuffled targets** for $N$ times to get $N$ feature random importances: $R_f = [r_{f_1},r_{f_2}...r_{f_N}]$.
3. Compute the final importances of a feature $f$ by various methods, such as:
    - $I_f = Avg(A_f) - Avg(R_f)$
    - $I_f = Avg(A_f) / (Avg(R_f) + 1)$

We want $M \ge 1$ and $N \gg 1$. Having $M=1$ means the actual importances depends on only 1 model's `random_state` (Which is also fine).

Not to be confused with [sklearn.inspection.permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance),
this sklearn method is about feature permutation instead of target permutation.

This method were originally proposed/implemented by:
- [[Paper] Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [[Kaggle Notebook] Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)

---

## Features
1. Compute null importances with only one function call.
2. Support models with `sklearn` interface, including `xgboost`, `catboost`, `lightgbm`.
3. Support data in `pandas.DataFrame` and `numpy.ndarray`
4. Highly customizable with both the exposed `compute` and `generic_compute` functions. 
5. Proven effectiveness in Kaggle competitions and in [`Our Benchmarks Results`](https://target-permutation-importances.readthedocs.io/en/latest/benchmarks/).

Here are some examples of Top Kaggle solutions using this method:

| Year | Competition                                                                                                                  | Medal | Link                                                                                                                                        |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play) | Gold  | [3rd place solution](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420235)                      |
| 2019 | [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/overview)    | Gold  | [16th place solution]([-play/discussion/420235](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82166)) |
| 2018 | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)                            | Gold  | [10th place solution](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64598)                                        |

Here is the summary of [`Our Benchmarks Results`](https://target-permutation-importances.readthedocs.io/en/latest/benchmarks/). It is running null-importances with feature selection on multiple models and datasets. "better" means it is better than running feature selection with the model's built-in feature importances. We can see even **with default models' parameters** it shows its effectiveness.

| model                  | n_dataset | n_better | better % |
|------------------------|-----------|----------|----------|
| CatBoostClassifier     | 10        | 6        | 60.0     |
| CatBoostRegressor      | 12        | 8        | 66.67    |
| LGBMClassifier         | 10        | 7        | 70.0     |
| LGBMRegressor          | 12        | 6        | 50.0     |
| RandomForestClassifier | 10        | 9        | 90.0     |
| RandomForestRegressor  | 12        | 7        | 58.33    |
| XGBClassifier          | 10        | 5        | 50.0     |
| XGBRegressor           | 12        | 5        | 41.67    |
---

## Install

```
pip install target-permutation-importances
```
or with poetry:
```
poetry add target-permutation-importances
```

Although this package is tested on models from `sklearn`, `xgboost`, `catboost`, `lightgbm`, they are not
a hard requirement for the installation, you can use this package for any model if it implements the `sklearn` interface.
For models that don't follow `sklearn` interface, you can use the exposed `generic_compute` method as discussed in the 
Advance Usage / Customization section.

Dependencies:
```
[tool.poetry.dependencies]
python = "^3.8"
numpy = "^1.21.0"
pandas = "^1.5.3"
tqdm = "^4.48.2"
beartype = "^0.14.1"
```
---

## Basic Usage

```python
# Import the function
import target_permutation_importances as tpi

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
result_df = tpi.compute(
    model_cls=RandomForestClassifier, # The constructor/class of the model.
    model_cls_params={ # The parameters to pass to the model constructor. Update this based on your needs.
        "n_jobs": -1,
    },
    model_fit_params={}, # The parameters to pass to the model fit method. Update this based on your needs.
    X=Xpd, # pd.DataFrame, np.ndarray
    y=data.target, # pd.Series, np.ndarray
    num_actual_runs=2,
    num_random_runs=10,
    # Options: {compute_permutation_importance_by_subtraction, compute_permutation_importance_by_division}
    # Or use your own function to calculate.
    permutation_importance_calculator=tpi.compute_permutation_importance_by_subtraction,
)

print(result_df[["feature", "importance"]].sort_values("importance", ascending=False).head())
```
Fork above code from [Kaggle](https://www.kaggle.com/code/kingychiu/target-permutation-importances-basic-usage/notebook).

Outputs:
```
Running 2 actual runs and 10 random runs
100%|██████████| 2/2 [00:01<00:00,  1.62it/s]
100%|██████████| 10/10 [00:06<00:00,  1.46it/s]
                 feature  importance
25       worst perimeter    0.117495
22  worst concave points    0.089949
26          worst radius    0.084632
7    mean concave points    0.064289
20            worst area    0.062485
8         mean concavity    0.047122
10        mean perimeter    0.029270
5              mean area    0.014566
11           mean radius    0.014346
0             area error    0.000693

```

You can find more detailed examples in the "Feature Selection Examples" section.

---

## Customization

**Changing model or parameters**

You can pick your own model by changing
`model_cls`, `model_cls_params` and `model_fit_params`, for example, using with `LGBMClassifier` 
with a `importance_type=gain` and `colsample_bytree=0.1`:

```python
result_df = tpi.compute(
    model_cls=LGBMClassifier, # The constructor/class of the model.
    model_cls_params={ # The parameters to pass to the model constructor. Update this based on your needs.
        "n_jobs": -1,
        "importance_type": "gain",
        "colsample_bytree": 0.1,
    },
    model_fit_params={}, # The parameters to pass to the model fit method. Update this based on your needs.
    X=Xpd, # pd.DataFrame, np.ndarray
    y=data.target, # pd.Series, np.ndarray
    num_actual_runs=2,
    num_random_runs=10,
    # Options: {compute_permutation_importance_by_subtraction, compute_permutation_importance_by_division}
    # Or use your own function to calculate.
    permutation_importance_calculator=tpi.compute_permutation_importance_by_subtraction,
)
```

**Changing null importances calculation**
You can pick your own calculation method by changing `permutation_importance_calculator`.
There are 2 provided calculations:
- `tpi.compute_permutation_importance_by_subtraction`
- `tpi.compute_permutation_importance_by_division`

You can also implement you own calculation function and pass it in. The function needs to follow 
`PermutationImportanceCalculatorType` specification, you can find it in
[API Reference](https://target-permutation-importances.readthedocs.io/en/latest/reference/)

**Advance Customization**

This package exposes `generic_compute` to allow advance customization.
Read [`target_permutation_importances.__init__.py`](https://github.com/kingychiu/target-permutation-importances/blob/main/target_permutation_importances/__init__.py) for details.

---

## Feature Selection Examples
- [Feature Selection for Binary Classification](https://www.kaggle.com/code/kingychiu/feature-selection-for-binary-classification-task)

---


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

---

